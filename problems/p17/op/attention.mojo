from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async

# =============================================================================
# Transformer Attention 메커니즘 구현
# =============================================================================
#
# Attention
# - "스마트한 정보 검색 기술" - 수많은 정보 중 현재 필요한 것만 선별적으로 추출
# - 수학적 정의: Attention(Q, K, V) = softmax(Q · K^T) · V
# - 실제 사용: 기계번역, 텍스트 생성, 이미지 인식 등 거의 모든 최신 AI 모델의 기반
#
# 도서관 비유로 이해하기:
# - Q (쿼리): "내가 찾고 있는 주제는 무엇인가?" (예: "인공지능의 역사")
# - K (키): "도서관에 어떤 책들이 있는가?" (예: 책 제목들)
# - V (값): "각 책의 실제 내용은 무엇인가?" (예: 책의 본문들)
# - 결과: 내 질문과 가장 관련 높은 책들의 내용을 중요도에 따라 조합한 정보
#
# 3단계 계산 과정 (구체적 예시: Q=[1,2], K=[[1,0],[0,1],[1,1]], V=[[10,20],[30,40],[50,60]]):
# 1. 유사도 점수 계산: Q·K^T → [1,2,3] (각 키와 쿼리의 관련성 점수)
# 2. 주의 가중치 생성: softmax([1,2,3]) → [0.09,0.245,0.665] (확률 분포로 변환)
# 3. 가중 평균 계산: 가중치·V → [41.5,51.5] (중요한 정보만 선별적으로 추출)
#
# 핵심 아이디어: 모든 정보를 동등하게 처리하지 않고, 현재 상황에서
# 가장 중요한 정보에만 '주의(Attention)'를 기울여 효율적으로 처리

alias SEQ_LEN = 16  # 시퀀스 길이 (키/값 벡터의 개수)
alias D = 16  # 벡터 차원 (각 벡터의 크기)
alias TPB = SEQ_LEN  # 스레드 블록 크기 (softmax 최적화를 위해 SEQ_LEN과 동일)


# =============================================================================
# 커널 재사용 전략: 검증된 부품들의 스마트한 조립
# =============================================================================
#
# 모듈러 GPU 아키텍처의 핵심 철학:
# - 복잡한 연산을 처음부터 새로 만들지 않음
# - 이미 검증되고 최적화된 기본 부품(커널)들을 조립
# - 각 커널은 단일 책임을 가지며 고도로 최적화됨
# - 재사용을 통해 개발 시간 단축 및 성능 향상
#
# Attention에서의 부품 활용 전략:
# 1. Q(1×d) @ K^T(d×seq_len) → Scores(1×seq_len): matmul_idiomatic_tiled 재사용
# 2. softmax(Scores) → Weights: softmax_kernel 재사용
# 3. Weights(1×seq_len) @ V(seq_len×d) → Output(1×d): matmul_idiomatic_tiled 재사용
#
# 이런 접근법의 장점:
# - 각 부품은 이미 성능이 검증됨 (Puzzle 14, 16에서 테스트됨)
# - 버그 발생 가능성 최소화
# - 유지보수 용이성
# - 확장성 (더 큰 어텐션으로 쉽게 확장 가능)


# =============================================================================
# 커널 재사용 전략: Puzzle 14의 Tiled Matrix Multiplication 활용
# =============================================================================
#
# 모듈러 GPU 아키텍처의 핵심 원칙:
# - 복잡한 연산을 검증된 기본 커널들의 조합으로 구현
# - 각 커널은 단일 책임을 가지며 고도로 최적화됨
# - 재사용을 통해 개발 시간 단축 및 성능 향상
#
# Attention에서의 매트릭스 곱셈 활용:
# 1. Q(1×d) @ K^T(d×seq_len) → Scores(1×seq_len)
# 2. Weights(1×seq_len) @ V(seq_len×d) → Output(1×d)
fn matmul_idiomatic_tiled[
    layout: Layout,
    rows: Int,
    cols: Int,
    inner: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    a: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
    b: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
):
    """Puzzle 14에서 가져온 최적화된 타일 기반 매트릭스 곱셈 커널."""
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + local_row
    tiled_col = block_idx.x * TPB + local_col

    # 현재 스레드 블록이 담당하는 출력 매트릭스의 타일
    out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc().fill(0)
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc().fill(0)

    var acc: output.element_type = 0

    alias load_a_layout = Layout.row_major(1, TPB)
    alias load_b_layout = Layout.row_major(TPB, 1)

    # 타일별로 순차 처리하여 전체 매트릭스 곱셈 수행
    for idx in range((inner + TPB - 1) // TPB):
        # A와 B 매트릭스에서 현재 처리할 타일 선택
        a_tile = a.tile[TPB, TPB](block_idx.y, idx)
        b_tile = b.tile[TPB, TPB](idx, block_idx.x)

        # =========================================================================
        # 비동기 메모리 복사 (Asynchronous Memory Copy)
        # =========================================================================
        #
        # copy_dram_to_sram_async 함수의 핵심 개념:
        # - DRAM (Device RAM): GPU의 글로벌 메모리 (느리지만 용량 크다)
        # - SRAM (Static RAM): GPU의 공유 메모리 (빠르지만 용량 작다)
        # - 비동기 복사: 메모리 복사와 다른 연산을 동시에 수행 가능
        #
        # 성능 최적화 원리:
        # 1. 메모리 대역폭 활용도 최대화
        # 2. 계산과 메모리 전송의 파이프라이닝
        # 3. 공유 메모리의 높은 대역폭 활용 (글로벌 메모리 대비 100배 빠름)
        #
        # thread_layout 매개변수의 의미:
        # - 어떤 스레드들이 메모리 복사에 참여할지 정의
        # - load_a_layout: 1×TPB 패턴 (행 방향 로딩)
        # - load_b_layout: TPB×1 패턴 (열 방향 로딩)
        # - 메모리 코얼레싱(coalescing) 최적화를 위한 패턴
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_shared, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_shared, b_tile)

        # =========================================================================
        # 비동기 복사 완료 대기 (Wait for Async Copy Completion)
        # =========================================================================
        #
        # async_copy_wait_all 함수의 필요성:
        # - 비동기 복사가 완료되기 전에 공유 메모리에 접근하면 데이터 레이스 발생
        # - 모든 pending 비동기 복사 작업이 완료될 때까지 대기
        # - GPU의 메모리 일관성(memory consistency) 보장
        #
        # 동작 원리:
        # 1. 하드웨어 레벨에서 비동기 복사 상태 추적
        # 2. 모든 복사 작업 완료 시그널 대기
        # 3. 메모리 펜스(memory fence) 역할로 순서 보장
        #
        # 성능 고려사항:
        # - 비동기 복사 중에는 다른 독립적인 연산 수행 가능
        # - 복사 완료 후에만 공유 메모리 데이터 사용 안전
        async_copy_wait_all()
        barrier()  # 모든 스레드가 데이터 로딩 완료까지 동기화

        # 현재 타일에 대한 부분 매트릭스 곱셈 수행
        @parameter
        for k in range(TPB):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()  # 다음 타일로 진행하기 전 동기화

    # Attention의 가변 크기 처리를 위한 경계 검사
    # (seq_len과 d가 TPB로 나누어떨어지지 않을 수 있음)
    if tiled_row < rows and tiled_col < cols:
        out_tile[local_row, local_col] = acc


# =============================================================================
# 공유 메모리 기반 매트릭스 전치 커널
# =============================================================================
#
# 전치가 필요한 이유:
# - Attention에서 K^T 계산 필수: Q(1×d) @ K^T(d×seq_len)
# - 원본 K는 (seq_len×d) 형태이므로 전치하여 (d×seq_len)로 변환
# - 구체적 예시: K[[1,0],[0,1],[1,1]] → K^T[[1,0,1],[0,1,1]]
#
# 공유 메모리 전치의 장점:
# 1. 메모리 코얼레싱: 읽기와 쓰기 모두 연속적 메모리 접근
# 2. 캐시 효율성: 공유 메모리를 임시 저장소로 활용 (글로벌 메모리 대비 100배 빠름)
# 3. 동기화 최소화: 블록 단위로 효율적 처리
#
# 핵심 아이디어: 2단계 스마트 전치
# 1단계: 글로벌 메모리 → 공유 메모리 (정상 인덱싱으로 로딩)
# 2단계: 공유 메모리 → 글로벌 메모리 (전치된 인덱싱으로 저장)
fn transpose_kernel[
    layout_in: Layout,  # 입력 매트릭스 레이아웃 (seq_len, d)
    layout_out: Layout,  # 출력 매트릭스 레이아웃 (d, seq_len)
    rows: Int,  # 입력 매트릭스의 행 수
    cols: Int,  # 입력 매트릭스의 열 수
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout_out, MutableAnyOrigin],
    inp: LayoutTensor[mut=False, dtype, layout_in, MutableAnyOrigin],
):
    # =========================================================================
    # 공유 메모리 타일링 전치 알고리즘
    # =========================================================================
    #
    # 전치 과정의 구체적 예시 (4×4 매트릭스):
    # 입력 매트릭스:     공유 메모리:        출력 매트릭스:
    # [1  2  3  4 ]     [1  2  3  4 ]     [1  5  9  13]
    # [5  6  7  8 ] →   [5  6  7  8 ] →   [2  6  10 14]
    # [9  10 11 12]     [9  10 11 12]     [3  7  11 15]
    # [13 14 15 16]     [13 14 15 16]     [4  8  12 16]
    #
    # 핵심: 공유 메모리에서 [row,col] → [col,row]로 인덱스 스왑

    # FILL ME IN (roughly 18 lines)
    shared_tile = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    local_row = thread_idx.y  # 블록 내 행 인덱스
    local_col = thread_idx.x  # 블록 내 열 인덱스

    global_row = block_idx.y * TPB + local_row  # 전체 매트릭스에서의 행 위치
    global_col = block_idx.x * TPB + local_col  # 전체 매트릭스에서의 열 위치

    # =========================================================================
    # 1단계: 입력 매트릭스에서 공유 메모리로 데이터 로딩
    # =========================================================================
    #
    # 예시 (Thread 0,0이 처리): inp[0,0] → shared_tile[0,0]
    # 예시 (Thread 1,2가 처리): inp[1,2] → shared_tile[1,2]
    # 경계 검사를 통해 유효한 데이터만 로딩
    if global_row < rows and global_col < cols:
        shared_tile[local_row, local_col] = inp[global_row, global_col]
    else:
        shared_tile[local_row, local_col] = 0.0  # 경계 밖은 0으로 패딩

    barrier()

    # =========================================================================
    # 2단계: 전치된 좌표 계산 및 출력
    # =========================================================================
    #
    # 블록 좌표도 전치: (block_idx.y, block_idx.x) → (block_idx.x, block_idx.y)
    # 이를 통해 매트릭스 차원 변환: (rows×cols) → (cols×rows)
    out_row = block_idx.x * TPB + local_row  # 전치된 행 위치
    out_col = block_idx.y * TPB + local_col  # 전치된 열 위치

    # =========================================================================
    # 전치의 핵심: 인덱스 스왑
    # =========================================================================
    #
    # 예시 전치 과정:
    # - Thread 0,0: shared_tile[0,0] → output[0,0] (대각선 요소)
    # - Thread 0,1: shared_tile[1,0] → output[0,1] (행↔열 스왑!)
    # - Thread 1,0: shared_tile[0,1] → output[1,0] (행↔열 스왑!)
    #
    # shared_tile[local_col, local_row] ← 여기서 전치 발생!
    # 정상: [row,col], 전치: [col,row]
    if out_row < cols and out_col < rows:
        output[out_row, out_col] = shared_tile[local_col, local_row]


# =============================================================================
# Puzzle 16의 Softmax 커널 재사용
# =============================================================================
#
# Attention Weights 생성을 위한 핵심 컴포넌트
# - 입력: 원시 주의 점수 (raw attention scores)
# - 출력: 정규화된 확률 분포 (∑weights = 1.0)
# - 수치적 안정성: exp(x - max(x)) 패턴 사용
fn softmax_gpu_kernel[
    layout: Layout,
    seq_len: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout, MutableAnyOrigin],
    scores: LayoutTensor[mut=False, dtype, layout, MutableAnyOrigin],
):
    """Puzzle 16의 검증된 softmax 구현을 그대로 활용."""
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # 1단계: 병렬 리덕션으로 최댓값 찾기
    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if global_i < seq_len:
        thread_max = rebind[Scalar[dtype]](scores[global_i])

    shared_max[local_i] = thread_max
    barrier()

    # 트리 기반 최댓값 리덕션
    stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared_max[local_i] = max(
                shared_max[local_i], shared_max[local_i + stride]
            )
        barrier()
        stride = stride // 2

    block_max = shared_max[0]

    # 2단계: 수치적 안정성을 위한 지수 계산
    var exp_val: Scalar[dtype] = 0.0
    if global_i < seq_len:
        exp_val = rebind[Scalar[dtype]](exp(scores[global_i] - block_max))
        output[global_i] = exp_val

    shared_sum[local_i] = exp_val
    barrier()

    # 3단계: 병렬 리덕션으로 지수 합계 계산
    stride = TPB // 2
    while stride > 0:
        if local_i < stride:
            shared_sum[local_i] = (
                shared_sum[local_i] + shared_sum[local_i + stride]
            )
        barrier()
        stride = stride // 2

    block_sum = shared_sum[0]

    # 4단계: 정규화로 확률 분포 생성
    if global_i < seq_len:
        output[global_i] = output[global_i] / block_sum


# =============================================================================
# CPU 버전: 벡터 어텐션 (Vector Attention)
# =============================================================================
#
# GPU 대비 CPU 구현의 특징:
# - 순차적 처리: 모든 연산을 하나씩 차례대로 수행
# - 단순한 반복문: 병렬 리덕션 대신 일반적인 for 루프
# - 직관적 구조: 수학적 정의를 그대로 코드로 구현
# - 디버깅 용이: 중간 결과 확인 및 검증에 적합
#
# 구체적 계산 예시로 이해하기:
# 입력: Q=[1,2], K=[[1,0],[0,1],[1,1]], V=[[10,20],[30,40],[50,60]]
# 결과: 약 [41.5, 51.5] (세 번째 값 벡터가 가장 큰 영향)
fn attention_cpu_kernel[
    layout_q: Layout,
    layout_k: Layout,
    layout_v: Layout,
    layout_out: Layout,
    seq_len: Int,
    d: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout_out, MutableAnyOrigin],
    q: LayoutTensor[dtype, layout_q, MutableAnyOrigin],
    k: LayoutTensor[dtype, layout_k, MutableAnyOrigin],
    v: LayoutTensor[dtype, layout_v, MutableAnyOrigin],
):
    """CPU에서 벡터 어텐션의 직관적 구현."""
    var scores = List[Float32]()
    var weights = List[Float32]()
    for _ in range(seq_len):
        scores.append(0.0)
        weights.append(0.0)

    # =========================================================================
    # 1단계: 어텐션 점수 계산 (Q · K[i])
    # =========================================================================
    #
    # 각 키 벡터 K[i]와 쿼리 Q의 내적 계산
    # 높은 점수 = 쿼리와 키의 유사도 높음 = 해당 값에 더 많은 주의
    #
    # 구체적 계산 예시 (Q=[1,2], K=[[1,0],[0,1],[1,1]]):
    # scores[0] = Q·K[0] = [1,2]·[1,0] = 1*1 + 2*0 = 1
    # scores[1] = Q·K[1] = [1,2]·[0,1] = 1*0 + 2*1 = 2
    # scores[2] = Q·K[2] = [1,2]·[1,1] = 1*1 + 2*1 = 3
    # 결과: scores = [1, 2, 3] (세 번째 키가 가장 관련성 높음)
    for i in range(seq_len):
        var score: Float32 = 0.0
        for dim in range(d):
            score = score + rebind[Float32](q[dim]) * rebind[Float32](k[i, dim])
        scores[i] = score

    # =========================================================================
    # 2단계: Softmax로 어텐션 가중치 생성
    # =========================================================================
    #
    # 수치적 안정성을 위한 3단계 softmax (앞의 scores=[1,2,3] 예시 계속):
    # 1) 최댓값 찾기: max([1,2,3]) = 3
    # 2) 안전한 지수 계산: exp([1-3, 2-3, 3-3]) = exp([-2,-1,0]) = [0.135, 0.368, 1.0]
    # 3) 정규화: 합계 1.503으로 나누기 → [0.09, 0.245, 0.665]
    # 결과: weights = [0.09, 0.245, 0.665] (합계=1.0, 세 번째가 가장 큰 가중치)
    var max_score: Float32 = scores[0]
    for i in range(1, seq_len):
        if scores[i] > max_score:
            max_score = scores[i]

    var sum_exp: Float32 = 0.0
    for i in range(seq_len):
        weights[i] = exp(scores[i] - max_score)
        sum_exp = sum_exp + weights[i]

    for i in range(seq_len):
        weights[i] = weights[i] / sum_exp

    # =========================================================================
    # 3단계: 가중 평균으로 최종 출력 계산
    # =========================================================================
    #
    # 각 값 벡터 V[i]에 해당 가중치를 곱한 후 모두 합산
    # 구체적 계산 (weights=[0.09,0.245,0.665], V=[[10,20],[30,40],[50,60]]):
    # 차원 0: 0.09*10 + 0.245*30 + 0.665*50 = 0.9 + 7.35 + 33.25 = 41.5
    # 차원 1: 0.09*20 + 0.245*40 + 0.665*60 = 1.8 + 9.8 + 39.9 = 51.5
    # 결과: output = [41.5, 51.5]
    #
    # 해석: 가중치가 가장 높은 세 번째 값 벡터 [50,60]이 최종 결과에 가장 큰 영향(66.5%)을 미침
    for dim in range(d):
        var weighted_sum: Float32 = 0.0
        for i in range(seq_len):
            weighted_sum = weighted_sum + weights[i] * rebind[Float32](
                v[i, dim]
            )
        output[dim] = rebind[Scalar[dtype]](weighted_sum)


# =============================================================================
# Attention 커스텀 연산 (Custom MAX Graph Operation)
# =============================================================================
#
# MAX Graph와의 통합:
# - 다중 입력 텐서 지원 (Q, K, V)
# - CPU/GPU 자동 선택
# - 메모리 최적화된 실행
# - Python과의 원활한 인터페이스
@compiler.register("attention")
struct AttentionCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # 실행 대상: "cpu" 또는 "gpu"
        seq_len: Int,  # 시퀀스 길이
        d: Int,  # 벡터 차원
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],  # 출력 벡터 (d,)
        q: InputTensor[rank=1],  # 쿼리 벡터 (d,)
        k: InputTensor[rank=2],  # 키 매트릭스 (seq_len, d)
        v: InputTensor[rank=2],  # 값 매트릭스 (seq_len, d)
        ctx: DeviceContextPtr,
    ) raises:
        # 레이아웃 정의 (메모리 구조 명시)
        alias layout_q = Layout.row_major(d)
        alias layout_k = Layout.row_major(seq_len, d)
        alias layout_v = Layout.row_major(seq_len, d)
        alias layout_out = Layout.row_major(d)
        alias layout_scores = Layout.row_major(seq_len)

        # 입력 텐서를 레이아웃 텐서로 변환
        var output_tensor = rebind[
            LayoutTensor[dtype, layout_out, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var q_tensor = rebind[LayoutTensor[dtype, layout_q, MutableAnyOrigin]](
            q.to_layout_tensor()
        )
        var k_tensor = rebind[LayoutTensor[dtype, layout_k, MutableAnyOrigin]](
            k.to_layout_tensor()
        )
        var v_tensor = rebind[LayoutTensor[dtype, layout_v, MutableAnyOrigin]](
            v.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            # ANCHOR: attention_orchestration
            # =========================================================================
            # GPU 어텐션 오케스트레이션 (Multi-Kernel Coordination)
            # =========================================================================
            #
            # 복잡한 어텐션 연산을 검증된 커널들의 조합으로 구현
            # 핵심 설계 원칙:
            # 1. 모듈성: 각 단계별로 전문화된 커널 사용
            # 2. 메모리 효율성: 최소한의 임시 버퍼로 최대 성능
            # 3. 재사용성: 이전 퍼즐의 검증된 구현 활용
            #
            # 7단계 GPU 어텐션 파이프라인:
            # 1. Q reshape (d,) → (1,d)
            # 2. K transpose (seq_len,d) → (d,seq_len)
            # 3. matmul: Q@K^T (1,d)@(d,seq_len) → scores(1,seq_len)
            # 4. scores reshape (1,seq_len) → (seq_len,)
            # 5. softmax: scores → weights (확률 분포)
            # 6. weights reshape (seq_len,) → (1,seq_len)
            # 7. matmul: weights@V (1,seq_len)@(seq_len,d) → output(1,d)
            var gpu_ctx = rebind[DeviceContext](ctx[])

            # 매트릭스 곱셈을 위한 레이아웃 정의
            alias layout_q_2d = Layout.row_major(1, d)  # Q를 (1, d)로 reshape
            alias layout_k_t = Layout.row_major(d, seq_len)  # K^T는 (d, seq_len)
            alias layout_scores_2d = Layout.row_major(
                1, seq_len
            )  # 점수 (1, seq_len)
            alias layout_weights_2d = Layout.row_major(
                1, seq_len
            )  # 가중치 (1, seq_len)
            alias layout_result_2d = Layout.row_major(1, d)  # 결과 (1, d)

            # GPU 블록 및 그리드 구성
            alias scores_blocks_per_grid = (
                (seq_len + TPB - 1) // TPB,
                (1 + TPB - 1) // TPB,
            )
            alias result_blocks_per_grid = (
                (d + TPB - 1) // TPB,
                (1 + TPB - 1) // TPB,
            )
            alias matmul_threads_per_block = (TPB, TPB)
            alias transpose_blocks_per_grid = (
                (seq_len + TPB - 1) // TPB,
                (d + TPB - 1) // TPB,
            )

            # =========================================================================
            # 메모리 최적화 전략: "스마트한 버퍼 재사용"
            # =========================================================================
            #
            # 일반적인 접근법 (비효율적):
            # - K^T용 버퍼, scores용 버퍼, weights용 버퍼, result용 버퍼 각각 할당
            # - 총 4개 버퍼 필요, 메모리 사용량 증가, 할당/해제 오버헤드
            #
            # 우리의 최적화 전략 (효율적):
            # - 단 2개 버퍼로 모든 연산 처리!
            # - k_t_buf: K^T 전용 (seq_len * d)
            # - scores_weights_buf: scores와 weights 공용 (seq_len)
            #
            # 핵심 기법들:
            # 1. Zero-copy reshape: 메모리 이동 없이 텐서 형태만 변경
            # 2. 버퍼 재사용: 동일한 메모리를 시간차로 다른 용도로 활용
            # 3. 최소 할당: 전체 복잡한 연산에 단 2개 버퍼만 사용
            #
            # 메모리 사용 타임라인:
            # - scores_weights_buf: scores 저장 → 계산 완료 → weights로 재해석
            # - 같은 메모리, 다른 목적! (시간적 분리로 충돌 방지)
            k_t_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len * d
            )  # K^T 저장용 (d, seq_len)
            scores_weights_buf = gpu_ctx.enqueue_create_buffer[dtype](
                seq_len
            )  # 점수와 가중치 공용

            k_t = LayoutTensor[mut=True, dtype, layout_k_t, MutableAnyOrigin](
                k_t_buf.unsafe_ptr()
            )

            # =========================================================================
            # 단계 1: Q 벡터를 매트릭스 곱셈용 2D 형태로 reshape
            # =========================================================================
            #
            # 왜 reshape가 필요한가?
            # - 매트릭스 곱셈 커널은 2D 입력을 기대함
            # - Q(d,) → Q(1,d): 벡터를 1×d 매트릭스로 재해석
            # - Zero-copy 연산: 메모리 복사 없이 텐서 해석 방식만 변경
            # - 이를 통해 검증된 matmul_idiomatic_tiled 커널 재사용 가능
            #
            # 구체적 예시: Q=[1,2] → Q=[[1,2]] (같은 데이터, 다른 모양)
            q_2d = q_tensor.reshape[layout_q_2d]()

            # =========================================================================
            # 단계 2: K 매트릭스 전치 (K → K^T)
            # =========================================================================
            #
            # 수학적 필요성: Q(1×d) @ K^T(d×seq_len) 계산을 위해
            # 원본 K(seq_len×d)를 K^T(d×seq_len)로 변환
            #
            # 구체적 예시:
            # K = [[1,0],[0,1],[1,1]] → K^T = [[1,0,1],[0,1,1]]
            #
            # 공유 메모리 기반 전치로 메모리 코얼레싱 최적화
            # - 읽기: 연속된 글로벌 메모리 → 공유 메모리
            # - 쓰기: 공유 메모리 → 연속된 글로벌 메모리 (전치됨)
            gpu_ctx.enqueue_function[
                transpose_kernel[layout_k, layout_k_t, seq_len, d, dtype]
            ](
                k_t,
                k_tensor,
                grid_dim=transpose_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            # =========================================================================
            # 단계 3: 어텐션 점수 계산 (Q @ K^T)
            # =========================================================================
            #
            # 핵심 연산: Q(1×d) @ K^T(d×seq_len) → Scores(1×seq_len)
            # 결과: 쿼리와 각 키 벡터 간의 유사도 점수
            #
            # 구체적 계산 (Q=[[1,2]], K^T=[[1,0,1],[0,1,1]]):
            # Scores = [[1,2]] @ [[1,0,1],[0,1,1]] = [[1*1+2*0, 1*0+2*1, 1*1+2*1]]
            #        = [[1, 2, 3]]
            #
            # Puzzle 14의 최적화된 타일 매트릭스 곱셈 커널 재사용
            # - 공유 메모리 타일링으로 메모리 대역폭 최적화
            # - 비동기 메모리 복사로 계산과 메모리 전송 파이프라이닝
            scores_2d = LayoutTensor[
                mut=True, dtype, layout_scores_2d, MutableAnyOrigin
            ](scores_weights_buf.unsafe_ptr())

            gpu_ctx.enqueue_function[
                matmul_idiomatic_tiled[layout_q_2d, 1, seq_len, d, dtype]
            ](
                scores_2d,
                q_2d,
                k_t,
                grid_dim=scores_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            # =========================================================================
            # 단계 4: Softmax용 텐서 reshape
            # =========================================================================
            #
            # 차원 변환: Scores(1×seq_len) → Weights(seq_len,)
            # 이유: Softmax 커널은 1D 입력을 기대하므로 형태 변환 필요
            #
            # 구체적 예시: [[1,2,3]] → [1,2,3] (같은 메모리, 다른 해석)
            # Zero-copy 재해석: 동일한 scores_weights_buf 메모리 재사용
            weights = scores_2d.reshape[layout_scores]()

            # =========================================================================
            # 단계 5: Softmax로 어텐션 가중치 생성
            # =========================================================================
            #
            # 핵심 변환: 원시 점수 → 정규화된 확률 분포
            #
            # 구체적 계산 (scores=[1,2,3] 계속):
            # 1) 최댓값: max([1,2,3]) = 3
            # 2) 안전한 지수: exp([1-3,2-3,3-3]) = exp([-2,-1,0]) = [0.135,0.368,1.0]
            # 3) 정규화: [0.135,0.368,1.0] / 1.503 = [0.09,0.245,0.665]
            # 결과: weights = [0.09,0.245,0.665] (합계=1.0)
            #
            # Puzzle 16의 검증된 수치적 안정성 보장 softmax 커널 사용
            # - 병렬 리덕션으로 최댓값과 합계 계산
            # - 공유 메모리 활용으로 스레드 간 효율적 협력
            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout_scores, seq_len, dtype]
            ](
                weights,
                weights,
                grid_dim=(1, 1),
                block_dim=(seq_len, 1),
            )

            # =========================================================================
            # 단계 6: 최종 매트릭스 곱셈용 가중치 reshape
            # =========================================================================
            #
            # 차원 변환: Weights(seq_len,) → Weights(1×seq_len)
            # 이유: 매트릭스 곱셈을 위해 다시 2D 형태로 변환
            #
            # 구체적 예시: [0.09,0.245,0.665] → [[0.09,0.245,0.665]]
            # 동일한 데이터, 다른 차원 해석 (zero-copy)
            weights_2d = weights.reshape[layout_weights_2d]()

            # =========================================================================
            # 단계 7: 가중 평균 계산 (Weights @ V)
            # =========================================================================
            #
            # 최종 어텐션 출력: Weights(1×seq_len) @ V(seq_len×d) → Result(1×d)
            #
            # 구체적 계산 (weights=[[0.09,0.245,0.665]], V=[[10,20],[30,40],[50,60]]):
            # Result = [[0.09,0.245,0.665]] @ [[10,20],[30,40],[50,60]]
            #        = [[0.09*10+0.245*30+0.665*50, 0.09*20+0.245*40+0.665*60]]
            #        = [[41.5, 51.5]]
            #
            # 해석: 가중치가 높은 값 벡터들이 최종 결과에 더 많이 기여
            # 세 번째 값 벡터 [50,60]이 66.5%의 영향력으로 결과를 주도
            #
            # 다시 한 번 Puzzle 14의 타일 매트릭스 곱셈 커널 재사용
            # - 동일한 최적화 기법 (공유 메모리, 비동기 복사) 적용
            result_2d = output_tensor.reshape[layout_result_2d]()

            gpu_ctx.enqueue_function[
                matmul_idiomatic_tiled[layout_weights_2d, 1, d, seq_len, dtype]
            ](
                result_2d,
                weights_2d,
                v_tensor,
                grid_dim=result_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            # 최종 결과는 result_2d에서 output_tensor로 자동 반영됨 (같은 메모리 공유)
            # Zero-copy 설계: reshape으로 (1,d) → (d,) 변환 (추가 복사 없음)
            # ANCHOR_END: attention_orchestration

        elif target == "cpu":
            # CPU 버전: 순차적이지만 직관적인 구현
            attention_cpu_kernel[
                layout_q, layout_k, layout_v, layout_out, seq_len, d, dtype
            ](output_tensor, q_tensor, k_tensor, v_tensor)

        else:
            raise Error("Unsupported target: " + target)
