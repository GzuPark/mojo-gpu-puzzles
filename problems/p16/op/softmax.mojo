from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite

# =============================================================================
# Softmax GPU 커널 구현
# =============================================================================
#
# Softmax란 무엇인가?
# - 여러 개의 숫자들을 받아서 총합이 1이 되는 확률 분포로 변환하는 함수
# - 예시: [2.0, 1.0, 0.1] → [0.659, 0.242, 0.099] (합계: 1.0)
# - 주용도: AI 모델이 여러 선택지 중 하나를 고를 때 각 선택지의 확률 계산
#
# 수치적 안정성 (Numerical Stability)
# - 문제: exp(x)는 x가 클 때 무한대로 발산 (오버플로우)
# - 해결: exp(x - max(x)) 사용으로 최대값을 0으로 만들어 안전하게 계산
# - 예시: [100, 101, 102] → exp([100-102, 101-102, 102-102]) = exp([-2, -1, 0])
#
# Softmax 계산 3단계:
# 1. 최댓값 찾기 (수치 안정성을 위해)
# 2. exp(입력값 - 최댓값) 계산 및 합계 구하기
# 3. 각 값을 합계로 나누어 확률 분포 생성

alias SIZE = 512
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)


# =============================================================================
# Buffer Reduction 전략을 위한 새로운 GPU 커널들
# =============================================================================
#
# 문제 정의: SIZE > TPB인 대규모 입력 처리
# - 기존: softmax_gpu_kernel은 SIZE ≤ TPB (128)에서만 효율적
# - 목표: SIZE=512, 1024, 2048 등 대규모 입력에서도 고성능 달성
#
# GPU Softmax 최적화 전략 비교
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 전략            │ 적용 조건          │ 장점              │ 단점            │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 1. Warp         │ K ≤ 32*pack_size │ 최고 성능         │ 제한된 크기       │
# │    Reduction    │ (작은 입력)       │ register 사용     │ 확장성 부족       │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 2. Block        │ 32 < K ≤ 1024    │ 중간 성능         │ shared mem 병목  │
# │    Reduction    │ (중간 입력)       │ 단일 커널         │ 메모리 제약       │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 3. Buffer       │ K > 1024         │ 무제한 확장성     │ 다중 커널 오버헤드  │
# │    Reduction    │ (대규모 입력)     │ 메모리 효율적      │ 복잡한 구현       │
# │    (현재 구현)   │                  │ 수치적 안정성     │                  │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ 4. Flash        │ 매우 큰 K        │ O(n) 메모리       │ 매우 복잡         │
# │    Attention    │ (LLM용)          │ 타일링 최적화     │ 구현 난이도 높음   │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ════════════════════════════════════════════════════════════════════════════
# 1. Warp Reduction 전략 (K ≤ 32*pack_size)
# ════════════════════════════════════════════════════════════════════════════
#
# 핵심 아이디어: GPU의 warp(32 스레드) 내에서 shuffle 명령어로 초고속 리덕션
#
# 메모리 계층 활용:
# - Register(레지스터): 각 스레드의 개인 저장소 - 가장 빠름
# - Shared Memory(공유 메모리): 블록 내 스레드 공유 - 빠름
# - Global Memory(글로벌 메모리): 모든 스레드 접근 - 느림
#
# Warp Shuffle 최적화:
# ```cuda
# // CUDA의 __shfl_xor_sync를 사용한 트리 리덕션
# for (int mask = 16; mask > 0; mask >>= 1) {
#     val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
# }
# ```
#
# 성능 특징:
# - 장점: 최고 성능 (shared memory 접근 없음), 지연 시간 최소
# - 단점: 크기 제한 (K ≤ 1024), 레지스터 압박
# - 적용: 작은 임베딩 차원, 분류 문제 최종 층
#
# ════════════════════════════════════════════════════════════════════════════
# 2. Block Reduction 전략 (32 < K ≤ 1024)
# ════════════════════════════════════════════════════════════════════════════
#
# 핵심 아이디어: 블록 내 모든 스레드가 shared memory를 통해 협력 리덕션
#
# 2단계 리덕션 구조:
# 1. Warp 내 리덕션: 각 warp(32스레드)가 독립적으로 리덕션 수행
# 2. Inter-warp 리덕션: warp 결과들을 shared memory로 취합
#
# 메모리 접근 패턴:
# ```
# 초기: Thread 0~1023이 각각 1개 원소 담당
# 1라운드: Thread 0~511이 인접 원소와 연산 (stride=512)
# 2라운드: Thread 0~255가 stride=256으로 연산
# ...
# 마지막: Thread 0이 최종 결과 계산
# ```
#
# 성능 특징:
# - 장점: 중간 성능, 단일 커널로 완성, 구현 상대적 간단
# - 단점: shared memory 병목, 블록 크기 제한(1024)
# - 적용: 중간 크기 시퀀스, 대부분의 실용적 케이스
#
# ════════════════════════════════════════════════════════════════════════════
# 3. Buffer Reduction 전략 (현재 구현: K > 1024)
# ════════════════════════════════════════════════════════════════════════════
#
# 핵심 아이디어: 연산을 3단계로 분해하여 무제한 확장성 달성
#
# 1단계: reduce_block_kernel
# - 입력을 GRID_SIZE개 청크로 분할 (각 청크 크기 = TPB)
# - 각 블록이 담당 청크에서 local_max, local_sum 계산
# - 결과를 중간 버퍼(block_maxes, block_sums)에 저장
#
# 예시 (SIZE=512, TPB=128)
# - Block 0: input[0:128]   -> local_max_0, local_sum_0
# - Block 1: input[128:256] -> local_max_1, local_sum_1
# - Block 2: input[256:384] -> local_max_2, local_sum_2
# - Block 3: input[384:512] -> local_max_3, local_sum_3
#
# 2단계: reduce_interim_kernel
# - 모든 블록의 중간 결과를 단일 블록에서 처리
# - global_max = max(local_max_0, local_max_1, ...)
# - global_sum = Σ(local_sum_i × exp(local_max_i - global_max))
# - 수치적 안정성을 위한 지수 보정 핵심 구현
#
# 수학적 정당성
# - exp(a-c) + exp(b-c) = exp(-c) × (exp(a) + exp(b))
# - 여기서 c = max(a,b)로 오버플로우 방지
#
# 3단계: normalize_kernel
# - 모든 입력 원소를 병렬로 최종 정규화
# - output[i] = exp(input[i] - global_max) / global_sum
# - 완벽한 병렬성으로 최대 처리량 달성
#
# 메모리 접근 패턴:
# - 순차적 읽기/쓰기로 메모리 대역폭 최대 활용
# - 캐시 지역성 최적화
#
# 성능 특징:
# - 장점: 무제한 확장성, 메모리 효율적, 수치적 안정성 보장
# - 단점: 3번의 커널 호출 오버헤드, 구현 복잡성
# - 적용: 대규모 시퀀스, Transformer 모델, 실시간 추론
#
# 메모리 사용량 분석 (SIZE=512 예시):
# - 입력/출력: 512 × sizeof(float32) = 2KB
# - 중간 버퍼: 4 × 2 × sizeof(float32) = 32B (매우 효율적!)
# - 총 추가 메모리: 1.6% (매우 경제적)
#
# ════════════════════════════════════════════════════════════════════════════
# 4. Flash Attention 전략 (초대규모 K > 10,000)
# ════════════════════════════════════════════════════════════════════════════
#
# 핵심 아이디어: HBM ↔ SRAM 간 데이터 이동 최소화로 메모리 벽 돌파
#
# 타일링(Tiling) 최적화:
# ```
# 전통적 방식: O(N²) 메모리 사용
# Q, K, V 전체 → attention_scores 전체 → softmax 전체
#
# Flash Attention: O(N) 메모리 사용
# Q, K, V 타일 → attention_scores 타일 → 점진적 softmax
# ```
#
# 점진적 Softmax 알고리즘:
# 1. 타일별 부분 계산: 각 타일에서 local_max, local_sum 계산
# 2. 온라인 업데이트: 글로벌 통계를 점진적으로 업데이트
# 3. 역정규화: 이전 결과를 새로운 글로벌 통계로 재조정
#
# 성능 혁신:
# - 메모리 복잡도: O(N²) → O(N)으로 선형 감소
# - 처리량: 기존 대비 2-4x 향상
# - 적용: GPT, BERT 등 대규모 언어 모델
#
# ════════════════════════════════════════════════════════════════════════════
# 5. Jagged Flash Attention (희소/가변 길이 데이터)
# ════════════════════════════════════════════════════════════════════════════
#
# 핵심 문제: 추천 시스템의 가변 길이 사용자 히스토리 처리
# ```
# 사용자 A: 100개 상호작용 → 패딩 불필요
# 사용자 B: 20개 상호작용  → 80개 패딩 (메모리 낭비)
# ```
#
# Jagged Tensor 구조:
# - Values: 모든 실제 데이터를 연속적으로 저장
# - Offsets: 각 샘플의 시작/끝 위치 표시
# - 패딩 완전 제거로 메모리 효율성 극대화
#
# 성능 개선:
# - 속도: 기존 대비 9x 향상
# - 메모리: 22x 절약
# - QPS: 10% 향상, 메모리 사용량 18% 감소
#
# ════════════════════════════════════════════════════════════════════════════
# 전략 선택 가이드라인
# ════════════════════════════════════════════════════════════════════════════
#
# 크기별 최적 전략:
# - K ≤ 32:          Warp Reduction (register shuffle)
# - 32 < K ≤ 1024:   Block Reduction (shared memory)
# - 1024 < K ≤ 10K:  Buffer Reduction (다중 커널) ← 현재 구현
# - K > 10K:         Flash Attention (타일링)
# - 가변 길이:        Jagged Flash Attention
#
# 용도별 추천:
# - 실시간 추론: Buffer Reduction (안정적 성능)
# - 대규모 훈련: Flash Attention (메모리 효율성)
# - 추천 시스템: Jagged Flash Attention (희소 데이터)
#
# 현재 구현의 위치:
# Buffer Reduction은 실용성과 성능의 최적 균형점
# - 복잡도: 중간 수준 (Flash Attention보다 단순)
# - 성능: 높음 (대부분 실제 사용 케이스에서 최적)
# - 확장성: 우수 (SIZE 무제한 확장 가능)
# - 안정성: 뛰어남 (수치적 정확성 보장)

# =============================================================================
# Buffer Reduction 전략을 위한 새로운 GPU 커널들
# =============================================================================


fn reduce_block_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    block_maxes: UnsafePointer[Scalar[dtype]],
    block_sums: UnsafePointer[Scalar[dtype]],
    input: LayoutTensor[mut=False, dtype, layout],
):
    # ═══════════════════════════════════════════════════════════════════════════
    # 1단계: Block-wise Reduction Kernel
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 목적: 대규모 입력을 블록 단위로 분할하여 각 블록의 local 통계 계산
    #
    # 핵심 전략:
    # 1. 데이터 분할: input[SIZE] → chunks[GRID_SIZE][TPB]
    # 2. 병렬 처리: 각 블록이 독립적으로 local_max, local_sum 계산
    # 3. 결과 저장: 중간 버퍼에 블록별 결과 저장
    #
    # 메모리 접근 패턴 최적화:
    # - Sequential Access: 각 스레드가 연속된 메모리 위치 접근
    # - Cache Locality: 블록 내 스레드들이 인접 데이터 처리
    # - Bandwidth Efficiency: coalesced memory access로 최대 처리량
    #
    # 예시 (SIZE=512, TPB=128, GRID_SIZE=4):
    # ```
    # Block 0: Thread 0~127 → input[0:128]   → local_max_0, local_sum_0
    # Block 1: Thread 0~127 → input[128:256] → local_max_1, local_sum_1
    # Block 2: Thread 0~127 → input[256:384] → local_max_2, local_sum_2
    # Block 3: Thread 0~127 → input[384:512] → local_max_3, local_sum_3
    # ```
    #
    # 수치적 안정성 고려사항:
    # - 각 블록에서 독립적으로 local_max 계산하여 지수 오버플로우 방지
    # - local_sum = Σ exp(x_i - local_max) 형태로 안전한 지수 계산
    # - 2단계에서 global_max 기준으로 재조정하여 최종 정확성 보장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📍 스레드 및 블록 인덱스 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    block_id = block_idx.x

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏠 공유 메모리 할당 - 블록 내 스레드 간 고속 통신
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 메모리 계층 구조 활용:
    # - Shared Memory: 블록 내 TPB(128)개 스레드가 공유
    # - 지연시간: ~100 clock cycles (Global Memory 대비 100x 빠름)
    # - 대역폭: ~1.5TB/s (GPU 모델에 따라 다름)
    #
    # 메모리 뱅크 최적화:
    # - 32개 뱅크로 구성된 shared memory에서 뱅크 충돌 회피
    # - 연속적 접근 패턴으로 최대 처리량 달성
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()  # 최댓값 리덕션용
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()  # 합계 리덕션용

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-1단계: 스레드별 입력 데이터 로드 및 초기 최댓값 설정
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 경계 처리 (Boundary Handling):
    # - 마지막 블록의 경우 input_size를 초과할 수 있음
    # - 초과 스레드들은 min_finite 값으로 초기화하여 최댓값 계산에 영향 없음
    #
    # 수치적 안정성 초기화:
    # - min_finite[dtype]: 해당 타입의 최소 유한값 (-∞에 가까움)
    # - 실제 데이터보다 항상 작은 값이므로 max 연산에서 안전
    var thread_max: Scalar[dtype] = min_finite[dtype]()

    if global_i < input_size:
        thread_max = rebind[Scalar[dtype]](input[global_i])

    shared_max[local_i] = thread_max
    barrier()  # 🚧 모든 스레드가 데이터 로드 완료할 때까지 동기화

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-2단계: 트리 기반 병렬 리덕션으로 블록 내 최댓값 찾기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # Binary Tree Reduction (O(log n) 복잡도)
    #
    # 단계별 리덕션 예시 (TPB=8일 때):
    # ```
    # 초기: [10, 1, 8, -1, 0, -2, 3, 5]
    #
    # stride=4: Thread 0~3 활성
    # Thread 0: max(10, 0) = 10    Thread 1: max(1, -2) = 1
    # Thread 2: max(8, 3) = 8      Thread 3: max(-1, 5) = 5
    # 결과: [10, 1, 8, 5, 0, -2, 3, 5]
    #
    # stride=2: Thread 0~1 활성
    # Thread 0: max(10, 8) = 10    Thread 1: max(1, 5) = 5
    # 결과: [10, 5, 8, 5, 0, -2, 3, 5]
    #
    # stride=1: Thread 0만 활성
    # Thread 0: max(10, 5) = 10
    # 최종: [10, 5, 8, 5, 0, -2, 3, 5] → block_max = 10
    # ```
    #
    # 성능 최적화 포인트:
    # - 활성 스레드 수 점진적 감소: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    # - 비활성 스레드는 자동으로 warp 레벨에서 최적화됨
    # - barrier() 호출로 데이터 레이스 조건 방지
    stride = TPB // 2  # 초기 stride: 64 (TPB=128인 경우)

    while stride > 0:
        var local_max: Scalar[dtype] = min_finite[dtype]()

        # 활성 스레드만 이웃 데이터와 비교
        if local_i < stride:
            local_max = rebind[Scalar[dtype]](shared_max[local_i + stride])

        barrier()

        # 활성 스레드만 최댓값 업데이트
        if local_i < stride:
            shared_max[local_i] = max(shared_max[local_i], local_max)

        barrier()
        stride = stride // 2  # 다음 라운드: 활성 스레드 수 절반으로 감소

    block_max = shared_max[0]  # 블록의 최댓값 (Thread 0에 저장됨)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-3단계: 수치적 안정성을 위한 지수 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 핵심 공식: exp(x_i - block_max)
    # - block_max 보정으로 지수 오버플로우 방지
    # - 각 블록에서 독립적으로 계산하여 수치적 안정성 확보
    #
    # 수학적 정당성 (Log-Sum-Exp Trick):
    # ```
    # 원래: exp(x_i) / Σexp(x_j)  ← 오버플로우 위험
    # 안전: exp(x_i - c) / Σexp(x_j - c)  ← c = max(x)로 정규화
    # ```
    #
    # 예시 (block_max = 10인 경우):
    # - input = [10, 8, 12, 6] → exp([0, -2, 2, -4]) = [1.0, 0.135, 7.389, 0.018]
    # - 모든 지수가 안전한 범위 내에서 계산됨
    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(input[global_i] - block_max))

    shared_sum[local_i] = exp_val
    barrier()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-4단계: 트리 기반 병렬 리덕션으로 블록 내 지수 합계 구하기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 목적: local_sum = Σ exp(x_i - block_max) 계산
    # - 최댓값 리덕션과 동일한 트리 구조 사용
    # - 합계 연산이므로 덧셈(+) 사용 (최댓값은 max 함수)
    #
    # 성능 특성:
    # - 시간 복잡도: O(log TPB) = O(log 128) = 7단계
    # - 공간 복잡도: O(TPB) shared memory 사용
    stride = TPB // 2
    while stride > 0:
        var local_sum: Scalar[dtype] = 0.0

        # 활성 스레드만 이웃 값과 합산
        if local_i < stride:
            local_sum = rebind[Scalar[dtype]](shared_sum[local_i + stride])

        barrier()

        # 활성 스레드만 합계 업데이트
        if local_i < stride:
            shared_sum[local_i] += local_sum

        barrier()
        stride = stride // 2

    block_sum = shared_sum[0]  # 블록의 지수 합계 (Thread 0에 저장됨)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-5단계: 블록별 결과를 중간 버퍼에 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 메모리 구조:
    # - block_maxes[GRID_SIZE]: 각 블록의 local_max 저장
    # - block_sums[GRID_SIZE]: 각 블록의 local_sum 저장
    #
    # 예시 (GRID_SIZE=4):
    # - block_maxes = [max_0, max_1, max_2, max_3]
    # - block_sums = [sum_0, sum_1, sum_2, sum_3]
    #
    # 최적화: Thread 0만 글로벌 메모리에 쓰기 (메모리 대역폭 절약)
    if local_i == 0:  # 블록의 대표 스레드만 저장
        block_maxes[block_id] = rebind[Scalar[dtype]](block_max)
        block_sums[block_id] = rebind[Scalar[dtype]](block_sum)


fn reduce_interim_kernel[
    dtype: DType = DType.float32,
](
    final_vals: UnsafePointer[Scalar[dtype]],  # [global_max, global_sum]
    block_maxes: UnsafePointer[Scalar[dtype]],
    block_sums: UnsafePointer[Scalar[dtype]],
    grid_size: Int,
):
    # ═══════════════════════════════════════════════════════════════════════════
    # 2단계: Global Aggregation Kernel
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 목적: 모든 블록의 중간 결과를 최종 글로벌 통계로 취합
    #
    # 핵심 도전과제: 수치적 안정성 유지하면서 블록 간 결과 합성
    # - 각 블록이 서로 다른 local_max를 가지므로 단순 합산 불가
    # - 글로벌 기준점(global_max)으로 모든 결과를 재조정 필요
    #
    # 입력 데이터 구조:
    # - block_maxes[GRID_SIZE]: [local_max_0, local_max_1, local_max_2, local_max_3]
    # - block_sums[GRID_SIZE]: [local_sum_0, local_sum_1, local_sum_2, local_sum_3]
    #
    # 출력 데이터 구조:
    # - final_vals[0]: global_max = max(block_maxes)
    # - final_vals[1]: global_sum = Σ(local_sum_i × exp(local_max_i - global_max))
    #
    # 수학적 정당성:
    # ```
    # 각 블록에서: local_sum_i = Σ exp(x_j - local_max_i)
    # 글로벌에서: global_sum = Σ_i Σ_j exp(x_j - global_max)
    #           = Σ_i (local_sum_i × exp(local_max_i - global_max))
    # ```
    #
    # 성능 특성:
    # - 단일 블록 실행 (grid_dim=(1,1))으로 동기화 오버헤드 최소화
    # - 입력 크기: GRID_SIZE (보통 4~32개) - 매우 작음
    # - 처리 시간: 전체 대비 1% 미만 (무시할 수 있는 수준)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 스레드 설정 및 메모리 할당
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    local_i = thread_idx.x

    # 공유 메모리 할당 (1단계와 동일한 패턴)
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-1단계: 글로벌 최댓값 찾기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 목적: global_max = max(local_max_0, local_max_1, ..., local_max_n)
    #
    # 예시 (GRID_SIZE=4):
    # - block_maxes = [10.5, 8.2, 12.1, 9.3]
    # - 트리 리덕션: 10.5 vs 8.2 → 10.5, 12.1 vs 9.3 → 12.1
    # - 최종: 10.5 vs 12.1 → global_max = 12.1
    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if local_i < grid_size:
        thread_max = block_maxes[local_i]

    shared_max[local_i] = thread_max
    barrier()

    # 트리 기반 병렬 리덕션으로 글로벌 최댓값 찾기 (1단계와 동일한 알고리즘)
    stride = TPB // 2
    while stride > 0:
        var local_max: Scalar[dtype] = min_finite[dtype]()

        if local_i < stride:
            local_max = rebind[Scalar[dtype]](shared_max[local_i + stride])

        barrier()

        if local_i < stride:
            shared_max[local_i] = max(shared_max[local_i], local_max)

        barrier()
        stride = stride // 2

    global_max = shared_max[0]  # 전체 데이터의 최댓값

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-2단계: 수치적 안정성을 위한 글로벌 합계 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 핵심 공식: global_sum = Σ (local_sum_i × exp(local_max_i - global_max))
    #
    # 수학적 배경:
    # ```
    # 원래 목표: Σ_i Σ_j exp(x_j - global_max)
    # 변형: Σ_i [ Σ_j exp(x_j - local_max_i) × exp(local_max_i - global_max) ]
    #     = Σ_i [ local_sum_i × exp(local_max_i - global_max) ]
    # ```
    #
    # 구체적 예시:
    # - Block 0: local_max=10.5, local_sum=25.3
    #   adjusted_sum_0 = 25.3 × exp(10.5 - 12.1) = 25.3 × 0.202 = 5.11
    # - Block 1: local_max=8.2, local_sum=18.7
    #   adjusted_sum_1 = 18.7 × exp(8.2 - 12.1) = 18.7 × 0.021 = 0.39
    # - Block 2: local_max=12.1, local_sum=42.8
    #   adjusted_sum_2 = 42.8 × exp(12.1 - 12.1) = 42.8 × 1.0 = 42.8
    # - Block 3: local_max=9.3, local_sum=31.2
    #   adjusted_sum_3 = 31.2 × exp(9.3 - 12.1) = 31.2 × 0.065 = 2.03
    #
    # 결과: global_sum = 5.11 + 0.39 + 42.8 + 2.03 = 50.33
    var adjusted_sum: Scalar[dtype] = 0.0
    if local_i < grid_size:
        adjusted_sum = block_sums[local_i] * rebind[Scalar[dtype]](
            exp(block_maxes[local_i] - global_max)
        )

    shared_sum[local_i] = adjusted_sum
    barrier()

    # 트리 기반 병렬 리덕션으로 글로벌 합계 구하기
    stride = TPB // 2
    while stride > 0:
        var local_sum: Scalar[dtype] = 0.0

        if local_i < stride:
            local_sum = rebind[Scalar[dtype]](shared_sum[local_i + stride])

        barrier()

        if local_i < stride:
            shared_sum[local_i] += local_sum

        barrier()
        stride = stride // 2

    global_sum = shared_sum[0]  # 전체 데이터의 지수 합계

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-3단계: 최종 글로벌 통계 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 출력 형식: final_vals[2] = [global_max, global_sum]
    # - 3단계 normalize_kernel에서 사용될 핵심 파라미터
    # - 모든 입력 데이터의 정규화 기준점 역할
    if local_i == 0:  # 대표 스레드만 최종 결과 저장
        final_vals[0] = rebind[Scalar[dtype]](global_max)
        final_vals[1] = rebind[Scalar[dtype]](global_sum)


fn normalize_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
    final_vals: UnsafePointer[Scalar[dtype]],  # [global_max, global_sum]
):
    # ═══════════════════════════════════════════════════════════════════════════
    # 3단계: Final Normalization Kernel
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 목적: 모든 입력을 최종 softmax 확률 분포로 변환
    #
    # 핵심 특징:
    # - 완벽한 병렬성: 각 스레드가 독립적으로 하나의 원소 처리
    # - 수치적 안정성: global_max 기준점 사용으로 오버플로우 방지
    # - 확률 분포 보장: 결과의 합이 정확히 1.0이 되도록 보장
    #
    # 성능 특성:
    # - 메모리 대역폭 한계: 대부분 메모리 읽기/쓰기 시간이 지배적
    # - 높은 병렬성: SIZE개 스레드가 동시 실행 가능
    # - 단순한 연산: 지수 함수와 나눗셈만 필요
    #
    # 입력/출력 패턴:
    # - 입력: 원본 데이터 input[SIZE], 글로벌 통계 final_vals[2]
    # - 출력: 정규화된 확률 분포 output[SIZE]
    # - 메모리 접근: coalesced 패턴으로 최대 대역폭 활용
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 최종 Softmax 확률 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # 공식: softmax(x_i) = exp(x_i - global_max) / global_sum
    #
    # 수학적 정당성:
    # ```
    # 표준 softmax: exp(x_i) / Σ exp(x_j)
    # 안전한 softmax: exp(x_i - c) / Σ exp(x_j - c)  (c = global_max)
    #
    # 증명: exp(x_i - c) / Σ exp(x_j - c)
    #     = [exp(-c) × exp(x_i)] / [exp(-c) × Σ exp(x_j)]
    #     = exp(x_i) / Σ exp(x_j)  (exp(-c) 상쇄)
    # ```
    #
    # 구체적 예시:
    # - input = [10.5, 8.2, 12.1, 9.3], global_max = 12.1, global_sum = 50.33
    #
    # - Thread 0: exp(10.5 - 12.1) / 50.33 = exp(-1.6) / 50.33 = 0.202 / 50.33 = 0.004
    # - Thread 1: exp(8.2 - 12.1) / 50.33 = exp(-3.9) / 50.33 = 0.020 / 50.33 = 0.0004
    # - Thread 2: exp(12.1 - 12.1) / 50.33 = exp(0) / 50.33 = 1.0 / 50.33 = 0.020
    # - Thread 3: exp(9.3 - 12.1) / 50.33 = exp(-2.8) / 50.33 = 0.061 / 50.33 = 0.001
    #
    # 검증: 0.004 + 0.0004 + 0.020 + 0.001 ≈ 1.000 ✓
    #
    # 경계 처리: global_i >= input_size인 스레드는 자동으로 무시됨
    if global_i < input_size:
        global_max = final_vals[0]  # 전체 데이터의 최댓값
        global_sum = final_vals[1]  # 전체 데이터의 지수 합계

        # 최종 softmax 확률 계산
        # - exp(input[global_i] - global_max): 수치적 안정성 보장
        # - / global_sum: 확률 분포 정규화 (합 = 1.0)
        output[global_i] = (
            rebind[Scalar[dtype]](exp(input[global_i] - global_max))
            / global_sum
        )


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # =========================================================================
    # 1단계: 병렬 리덕션으로 최댓값 찾기
    # =========================================================================
    #
    # 공유 메모리 (Shared Memory) 사용 이유:
    # - 모든 스레드가 협력하여 최댓값과 합계를 계산해야 함
    # - 글로벌 메모리보다 100배 이상 빠른 접근 속도
    # - GPU 블록 내 스레드들 간의 효율적인 데이터 공유
    #
    # 예시: TPB=8일 때, 8개 스레드가 각각 하나의 입력값을 담당
    # Thread 0: input[0], Thread 1: input[1], ..., Thread 7: input[7]
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    # 각 스레드가 자신이 담당하는 입력값을 공유 메모리에 저장
    var thread_max: Scalar[dtype] = min_finite[dtype]()

    if global_i < input_size:
        thread_max = rebind[Scalar[dtype]](input[global_i])

    shared_max[local_i] = thread_max

    barrier()  # 모든 스레드가 값을 저장할 때까지 대기

    # =========================================================================
    # 트리 기반 병렬 리덕션 (Tree-based Parallel Reduction)
    # =========================================================================
    #
    # 토너먼트 방식으로 최댓값 찾기 - O(log n) 시간 복잡도
    #
    # 구체적 예시 (TPB=8, 입력: [3, 1, 4, 1, 5, 9, 2, 6]):
    #
    # 초기상태: shared_max = [3, 1, 4, 1, 5, 9, 2, 6]
    #
    # 1라운드 (stride=4):
    #   Thread 0: max(3, 5) = 5  →  shared_max[0] = 5
    #   Thread 1: max(1, 9) = 9  →  shared_max[1] = 9
    #   Thread 2: max(4, 2) = 4  →  shared_max[2] = 4
    #   Thread 3: max(1, 6) = 6  →  shared_max[3] = 6
    #   결과: shared_max = [5, 9, 4, 6, 5, 9, 2, 6]
    #
    # 2라운드 (stride=2):
    #   Thread 0: max(5, 4) = 5  →  shared_max[0] = 5
    #   Thread 1: max(9, 6) = 9  →  shared_max[1] = 9
    #   결과: shared_max = [5, 9, 4, 6, 5, 9, 2, 6]
    #
    # 3라운드 (stride=1):
    #   Thread 0: max(5, 9) = 9  →  shared_max[0] = 9
    #   최종 결과: 전체 최댓값 = 9
    stride = TPB // 2

    while stride > 0:
        var local_max: Scalar[dtype] = min_finite[dtype]()

        if local_i < stride:
            local_max = rebind[Scalar[dtype]](shared_max[local_i + stride])

        barrier()

        if local_i < stride:
            shared_max[local_i] = max(shared_max[local_i], local_max)

        barrier()

        stride = stride // 2

    block_max = shared_max[0]  # 전체 블록의 최댓값

    # =========================================================================
    # 2단계: 수치적 안정성을 위한 지수 계산
    # =========================================================================
    #
    # 수학적 정당성과 구체적 예시:
    # 원래 공식: softmax(x_i) = exp(x_i) / Σexp(x_j)
    # 안전한 공식: softmax(x_i) = exp(x_i - c) / Σexp(x_j - c)  (c = max(x))
    #
    # 예시: 입력 [100, 101, 102], max = 102
    # - 원래: exp(100), exp(101), exp(102) → 모두 거대한 수 (오버플로우)
    # - 안전: exp(-2), exp(-1), exp(0) = [0.135, 0.368, 1.0] → 안전한 범위
    #
    # 중요: 수학적으로 완전히 동일한 결과를 보장하면서 오버플로우만 방지
    var exp_val: Scalar[dtype] = 0.0

    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(input[global_i] - block_max))
        output[global_i] = exp_val  # 임시로 지수값 저장

    shared_sum[local_i] = exp_val

    barrier()

    # =========================================================================
    # 3단계: 합계를 위한 두 번째 병렬 리덕션
    # =========================================================================
    #
    # Softmax 특성상 두 단계 리덕션이 필요한 이유:
    # 1단계: 최댓값 구하기 (수치 안정성을 위해 먼저 필요)
    # 2단계: 지수의 합 구하기 (정규화를 위해 두 번째로 필요)
    #
    # 구체적 예시 (앞의 지수값들 [0.135, 0.368, 1.0, ...]의 합 구하기):
    #
    # 1라운드: 인접한 값들끼리 더하기
    #   Thread 0: 0.135 + next_value
    #   Thread 1: 0.368 + next_value
    #   ...
    # 최종: 모든 지수값의 합계 (예: 3.5)
    stride = TPB // 2

    while stride > 0:
        var local_sum: Scalar[dtype] = 0.0

        if local_i < stride:
            local_sum = rebind[Scalar[dtype]](shared_sum[local_i + stride])

        barrier()

        if local_i < stride:
            shared_sum[local_i] += local_sum

        barrier()

        stride = stride // 2

    block_sum = shared_sum[0]  # 모든 지수값의 합계

    # =========================================================================
    # 4단계: 최종 확률 분포 생성 (정규화)
    # =========================================================================
    #
    # 확률 분포로의 변환:
    # - 이전: 단순 지수값 [0.135, 0.368, 1.0] (합계: 1.503)
    # - 현재: 확률값 [0.090, 0.245, 0.665] (합계: 1.0)
    #
    # 정규화 공식: softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)
    # 결과: 모든 값이 0~1 사이, 전체 합이 정확히 1.0인 확률 분포
    if global_i < input_size:
        output[global_i] = output[global_i] / block_sum


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    # =========================================================================
    # CPU vs GPU 구현 방식의 핵심 차이점
    # =========================================================================
    #
    # CPU 버전 특징:
    # - 순차적 처리: 하나씩 차례대로 계산 (O(n) 시간 복잡도)
    # - 단순한 for 루프 사용
    # - 지역 변수로 임시 값 저장
    # - 이해하기 쉬운 직관적 구조
    #
    # GPU 버전 특징:
    # - 병렬 처리: 여러 스레드가 동시에 계산 (O(log n) 시간 복잡도)
    # - 복잡한 병렬 리덕션 알고리즘
    # - 공유 메모리로 스레드 간 협력
    # - 높은 성능, 복잡한 구조
    #
    # 구체적 예시 비교 (입력: [2.0, 1.0, 0.1]):
    # CPU: 스레드 1개가 순차적으로 모든 계산 수행
    # GPU: 스레드 3개가 동시에 각자의 값을 처리하며 협력

    # =========================================================================
    # 1단계: 최댓값 찾기 (수치 안정성)
    # =========================================================================
    #
    # 예시: 입력 [2.0, 1.0, 0.1]에서 최댓값 2.0 찾기
    # - 초기값: min_finite (가장 작은 가능한 값)
    # - 순차 비교: min_finite → 2.0 → 2.0 → 2.0 (최종)
    var max_val: Scalar[dtype] = min_finite[dtype]()

    for i in range(input_size):
        max_val = max(max_val, rebind[Scalar[dtype]](input[i]))

    # =========================================================================
    # 2단계: 지수 계산 및 합계 구하기
    # =========================================================================
    #
    # 예시 계산 과정 (입력: [2.0, 1.0, 0.1], max_val: 2.0):
    # i=0: exp(2.0 - 2.0) = exp(0.0) = 1.0,     sum_exp = 1.0
    # i=1: exp(1.0 - 2.0) = exp(-1.0) = 0.368,  sum_exp = 1.368
    # i=2: exp(0.1 - 2.0) = exp(-1.9) = 0.150,  sum_exp = 1.518
    #
    # 임시 결과: output = [1.0, 0.368, 0.150], sum_exp = 1.518
    var sum_exp: Scalar[dtype] = 0.0

    for i in range(input_size):
        var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
        output[i] = exp_val
        sum_exp += exp_val

    # =========================================================================
    # 3단계: 정규화 (확률 분포 생성)
    # =========================================================================
    #
    # 최종 확률 계산 (앞의 예시 계속):
    # i=0: 1.0 / 1.518 = 0.659     (65.9% 확률)
    # i=1: 0.368 / 1.518 = 0.242   (24.2% 확률)
    # i=2: 0.150 / 1.518 = 0.099   (9.9% 확률)
    #
    # 검증: 0.659 + 0.242 + 0.099 = 1.0 ✓
    # 결과: 가장 큰 입력값(2.0)이 가장 높은 확률(65.9%)을 가짐
    for i in range(input_size):
        output[i] = output[i] / sum_exp


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())

        alias layout = input_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            var GRID_SIZE = (input_size + TPB - 1) // TPB

            # 중간 결과를 저장할 디바이스 버퍼 할당
            var block_maxes_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                GRID_SIZE
            )
            var block_sums_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                GRID_SIZE
            )
            var final_vals_buffer = gpu_ctx.enqueue_create_buffer[dtype](2)

            # GPU 메모리 초기화의 중요성
            # 이전: 단순 연산에서는 초기화 불필요
            # 현재: 누적 연산(합계)이 있어서 초기값이 0이어야 함
            #
            # Softmax에서 초기화가 중요한 이유:
            # 1. 이전 실행의 잔여 데이터가 결과에 영향을 줄 수 있음
            # 2. 병렬 리덕션에서 잘못된 초기값은 전체 결과를 오염시킴
            # 3. 확률 분포의 정확성을 보장하기 위해 필수
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            # GPU 메모리 초기화의 중요성
            # 이전: 단순 연산에서는 초기화 불필요
            # 현재: 누적 연산(합계)이 있어서 초기값이 0이어야 함
            #
            # Buffer reduction에서 초기화가 더욱 중요한 이유:
            # 1. 이전 실행의 잔여 데이터가 중간 버퍼에 영향을 줄 수 있음
            # 2. 병렬 리덕션에서 잘못된 초기값은 전체 결과를 오염시킴
            # 3. 다단계 파이프라인에서 각 단계의 정확성을 보장하기 위해 필수
            #
            # 각 중간 버퍼를 0으로 초기화:
            # - block_maxes_buffer: 최댓값 계산용 (안전한 초기값)
            # - block_sums_buffer: 합계 계산용 (반드시 0이어야 함)
            # - final_vals_buffer: 최종 결과 저장용 (깨끗한 상태)
            gpu_ctx.enqueue_memset(block_maxes_buffer, 0.0)
            gpu_ctx.enqueue_memset(block_sums_buffer, 0.0)
            gpu_ctx.enqueue_memset(final_vals_buffer, 0.0)

            # 1단계: 블록 단위 리덕션 커널 실행
            gpu_ctx.enqueue_function[
                reduce_block_kernel[layout, input_size, dtype]
            ](
                block_maxes_buffer.unsafe_ptr(),
                block_sums_buffer.unsafe_ptr(),
                input_tensor,
                grid_dim=(GRID_SIZE, 1),
                block_dim=THREADS_PER_BLOCK,
            )

            # 2단계: 중간 결과 리덕션 커널 실행
            gpu_ctx.enqueue_function[reduce_interim_kernel[dtype]](
                final_vals_buffer.unsafe_ptr(),
                block_maxes_buffer.unsafe_ptr(),
                block_sums_buffer.unsafe_ptr(),
                GRID_SIZE,
                grid_dim=(1, 1),
                block_dim=THREADS_PER_BLOCK,
            )

            # 3단계: 최종 정규화 커널 실행
            gpu_ctx.enqueue_function[
                normalize_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                final_vals_buffer.unsafe_ptr(),
                grid_dim=(GRID_SIZE, 1),
                block_dim=THREADS_PER_BLOCK,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
