from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite

# =============================================================================
# Batched Softmax GPU 커널 구현
# =============================================================================
#
# Batched Softmax란 무엇인가?
# - 2D 입력 텐서의 각 행에 대해 독립적으로 softmax를 적용하는 함수
# - 예시: [[2.0, 1.0], [3.0, 0.1]] → [[0.731, 0.269], [0.952, 0.048]]
# - 주용도: 배치 단위로 여러 샘플을 동시에 처리하는 AI 모델
#
# Row-wise vs Column-wise Processing
# - Row-wise: 각 행(배치 샘플)에 대해 독립적으로 softmax 적용
# - Column-wise: 각 열(특성 차원)에 대해 독립적으로 softmax 적용
# - 현재 구현: Row-wise (가장 일반적인 사용 패턴)

# 2D Batched Softmax 설정
alias BATCH_SIZE = 8  # 배치 크기 (행 개수)
alias FEATURE_SIZE = 512  # 특성 크기 (열 개수)
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(BATCH_SIZE, FEATURE_SIZE)

# =============================================================================
# Buffer Reduction 전략을 위한 새로운 GPU 커널들 (2D Batched 버전)
# =============================================================================
#
# 문제 정의: 2D 입력 (BATCH_SIZE, FEATURE_SIZE)에서 각 행별 독립 처리
# - 기존: 1D 벡터 (SIZE=512)에 대한 Buffer Reduction
# - 목표: 2D 텐서 (2, 256)에서 각 행별로 softmax 적용
#
# 2D Batched Softmax 최적화 전략
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ 처리 방식       │ 메모리 레이아웃    │ 장점              │ 단점              │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ Row-wise       │ 연속된 메모리     │ 캐시 지역성 우수    │ 행별 동기화 필요   │
# │ (현재 구현)     │ 접근 패턴         │ 메모리 효율적      │                  │
# ├─────────────────────────────────────────────────────────────────────────┤
# │ Column-wise    │ 스트라이드 접근    │ 병렬성 극대화      │ 캐시 미스 증가     │
# │                │ 패턴             │                  │ 메모리 비효율적     │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ════════════════════════════════════════════════════════════════════════════
# 1단계: reduce_block_kernel_2d - 각 행별 블록 단위 리덕션
# ════════════════════════════════════════════════════════════════════════════
#
# 핵심 아이디어: 각 행을 독립적으로 처리하되, 행 내에서는 기존 Buffer Reduction 적용
#
# 메모리 접근 패턴 (BATCH_SIZE=4, FEATURE_SIZE=128 예시):
# - Row 0: input[0, 0:128]   -> local_max_0, local_sum_0
# - Row 1: input[1, 0:128]   -> local_max_1, local_sum_1
# - Row 2: input[2, 0:128]   -> local_max_2, local_sum_2
# - Row 3: input[3, 0:128]   -> local_max_3, local_sum_3
#
# 각 행은 독립적으로 처리되므로 완벽한 병렬성 달성
# 행 내에서는 FEATURE_SIZE > TPB인 경우 Buffer Reduction 적용

# =============================================================================
# 2D Batched Softmax를 위한 새로운 GPU 커널들
# =============================================================================


fn reduce_block_kernel_2d[
    layout: Layout,
    batch_size: Int,
    feature_size: Int,
    dtype: DType = DType.float32,
](
    block_maxes: UnsafePointer[Scalar[dtype]],  # [batch_size * blocks_per_row]
    block_sums: UnsafePointer[Scalar[dtype]],  # [batch_size * blocks_per_row]
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    # ═══════════════════════════════════════════════════════════════════════════
    # 2D Batched: 1단계 - 각 행별 블록 단위 리덕션
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 핵심 아이디어: 각 행을 독립적으로 처리하되, 행 내에서는 Buffer Reduction 적용
    #
    # 메모리 구조 (BATCH_SIZE=4, FEATURE_SIZE=128, TPB=128 예시):
    # - Row 0: input[0, 0:128] → block_maxes[0], block_sums[0]
    # - Row 1: input[1, 0:128] → block_maxes[1], block_sums[1]
    # - Row 2: input[2, 0:128] → block_maxes[2], block_sums[2]
    # - Row 3: input[3, 0:128] → block_maxes[3], block_sums[3]
    #
    # FEATURE_SIZE > TPB인 경우:
    # - 각 행을 여러 블록으로 분할하여 처리
    # - blocks_per_row = (FEATURE_SIZE + TPB - 1) // TPB

    local_i = thread_idx.x
    block_id = block_idx.x

    # 2D 인덱싱: 어떤 행(batch)과 어떤 블록(feature chunk)인지 계산
    blocks_per_row = (feature_size + TPB - 1) // TPB
    batch_idx = block_id // blocks_per_row  # 현재 처리 중인 배치 인덱스
    feature_block_idx = block_id % blocks_per_row  # 현재 행 내 블록 인덱스

    # 경계 체크: 유효한 배치 범위 내에서만 처리
    if batch_idx >= batch_size:
        return

    # 현재 스레드가 담당할 전역 인덱스 계산
    global_feature_idx = feature_block_idx * TPB + local_i

    # 공유 메모리 할당 (기존 1D와 동일한 패턴)
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-1단계: 각 행 내에서 블록별 최댓값 찾기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if global_feature_idx < feature_size:
        # 2D 인덱싱: input[batch_idx, global_feature_idx]
        thread_max = rebind[Scalar[dtype]](input[batch_idx, global_feature_idx])

    shared_max[local_i] = thread_max
    barrier()

    # 트리 기반 병렬 리덕션으로 블록 내 최댓값 찾기
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

    var block_max = shared_max[0]  # 현재 블록의 최댓값

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-2단계: 지수 계산 및 블록별 합계 구하기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var exp_val: Scalar[dtype] = 0.0
    if global_feature_idx < feature_size:
        exp_val = rebind[Scalar[dtype]](
            exp(input[batch_idx, global_feature_idx] - block_max)
        )

    shared_sum[local_i] = exp_val
    barrier()

    # 트리 기반 병렬 리덕션으로 블록 내 지수 합계 구하기
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

    var block_sum = shared_sum[0]  # 현재 블록의 지수 합계

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-3단계: 블록별 결과를 중간 버퍼에 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if local_i == 0:  # 블록의 대표 스레드만 저장
        block_maxes[block_id] = rebind[Scalar[dtype]](block_max)
        block_sums[block_id] = rebind[Scalar[dtype]](block_sum)


fn reduce_interim_kernel_2d[
    dtype: DType = DType.float32,
](
    final_vals: UnsafePointer[
        Scalar[dtype]
    ],  # [batch_size * 2] (각 행의 [global_max, global_sum])
    block_maxes: UnsafePointer[Scalar[dtype]],
    block_sums: UnsafePointer[Scalar[dtype]],
    batch_size: Int,
    blocks_per_row: Int,
):
    # ═══════════════════════════════════════════════════════════════════════════
    # 2D Batched: 2단계 - 각 행별 글로벌 통계 계산
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 핵심: 각 행에 대해 독립적으로 글로벌 max와 sum 계산
    # - final_vals 구조: [row0_max, row0_sum, row1_max, row1_sum, ...]

    local_i = thread_idx.x
    batch_idx = block_idx.x  # 각 블록이 하나의 배치(행)를 담당

    # 경계 체크
    if batch_idx >= batch_size:
        return

    # 공유 메모리 할당
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-1단계: 현재 행의 글로벌 최댓값 찾기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if local_i < blocks_per_row:
        # 현재 행의 block_idx 계산
        row_block_idx = batch_idx * blocks_per_row + local_i
        thread_max = block_maxes[row_block_idx]

    shared_max[local_i] = thread_max
    barrier()

    # 트리 리덕션으로 행별 글로벌 최댓값 찾기
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

    var global_max = shared_max[0]  # 현재 행의 글로벌 최댓값

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-2단계: 현재 행의 글로벌 합계 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var adjusted_sum: Scalar[dtype] = 0.0
    if local_i < blocks_per_row:
        row_block_idx = batch_idx * blocks_per_row + local_i
        adjusted_sum = block_sums[row_block_idx] * rebind[Scalar[dtype]](
            exp(block_maxes[row_block_idx] - global_max)
        )

    shared_sum[local_i] = adjusted_sum
    barrier()

    # 트리 리덕션으로 행별 글로벌 합계 구하기
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

    global_sum = shared_sum[0]  # 현재 행의 글로벌 합계

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-3단계: 행별 결과를 최종 버퍼에 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if local_i == 0:
        final_vals[batch_idx * 2] = rebind[Scalar[dtype]](
            global_max
        )  # [batch_idx][0]
        final_vals[batch_idx * 2 + 1] = rebind[Scalar[dtype]](
            global_sum
        )  # [batch_idx][1]


fn normalize_kernel_2d[
    layout: Layout,
    batch_size: Int,
    feature_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
    final_vals: UnsafePointer[Scalar[dtype]],  # [batch_size * 2]
):
    # ═══════════════════════════════════════════════════════════════════════════
    # 2D Batched: 3단계 - 각 행별 최종 정규화
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 핵심: 각 행에 대해 독립적으로 최종 softmax 정규화 수행
    # output[batch_idx, feature_idx] = exp(input[batch_idx, feature_idx] - global_max) / global_sum

    local_i = thread_idx.x
    block_id = block_idx.x

    # 2D 인덱싱: 어떤 행(batch)과 어떤 블록(feature chunk)인지 계산
    blocks_per_row = (feature_size + TPB - 1) // TPB
    batch_idx = block_id // blocks_per_row
    feature_block_idx = block_id % blocks_per_row

    # 경계 체크
    if batch_idx >= batch_size:
        return

    # 현재 스레드가 담당할 전역 특성 인덱스
    global_feature_idx = feature_block_idx * TPB + local_i

    # 경계 체크: 유효한 특성 범위 내에서만 처리
    if global_feature_idx >= feature_size:
        return

    # 현재 행의 글로벌 통계 가져오기
    global_max = final_vals[batch_idx * 2]  # 현재 행의 global_max
    global_sum = final_vals[batch_idx * 2 + 1]  # 현재 행의 global_sum

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3단계: 최종 softmax 정규화
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공식: output[i,j] = exp(input[i,j] - global_max_i) / global_sum_i
    # - 각 행(i)별로 독립적인 global_max_i, global_sum_i 사용
    # - 완벽한 병렬 처리: 모든 원소가 동시에 계산됨
    output[batch_idx, global_feature_idx] = rebind[Scalar[dtype]](
        exp(input[batch_idx, global_feature_idx] - global_max) / global_sum
    )


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


# =============================================================================
# Column-wise Softmax를 위한 새로운 GPU 커널들
# =============================================================================


fn reduce_block_kernel_2d_colwise[
    layout: Layout,
    batch_size: Int,
    feature_size: Int,
    dtype: DType = DType.float32,
](
    block_maxes: UnsafePointer[
        Scalar[dtype]
    ],  # [feature_size * blocks_per_col]
    block_sums: UnsafePointer[Scalar[dtype]],  # [feature_size * blocks_per_col]
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    # ═══════════════════════════════════════════════════════════════════════════
    # Column-wise: 1단계 - 각 열별 블록 단위 리덕션
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 핵심 아이디어: 각 열을 독립적으로 처리하되, 열 내에서는 Buffer Reduction 적용
    #
    # 메모리 구조 (BATCH_SIZE=2, FEATURE_SIZE=256, TPB=128 예시):
    # - Col 0: input[0:2, 0] → block_maxes[0], block_sums[0]
    # - Col 1: input[0:2, 1] → block_maxes[1], block_sums[1]
    # - ...
    # - Col 255: input[0:2, 255] → block_maxes[255], block_sums[255]
    #
    # BATCH_SIZE > TPB인 경우:
    # - 각 열을 여러 블록으로 분할하여 처리
    # - blocks_per_col = (BATCH_SIZE + TPB - 1) // TPB

    local_i = thread_idx.x
    block_id = block_idx.x

    # Column-wise 인덱싱: 어떤 열(feature)과 어떤 블록(batch chunk)인지 계산
    blocks_per_col = (batch_size + TPB - 1) // TPB
    feature_idx = block_id // blocks_per_col  # 현재 처리 중인 특성 인덱스
    batch_block_idx = block_id % blocks_per_col  # 현재 열 내 블록 인덱스

    # 경계 체크: 유효한 특성 범위 내에서만 처리
    if feature_idx >= feature_size:
        return

    # 현재 스레드가 담당할 전역 배치 인덱스 계산
    global_batch_idx = batch_block_idx * TPB + local_i

    # 공유 메모리 할당
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-1단계: 각 열 내에서 블록별 최댓값 찾기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if global_batch_idx < batch_size:
        # Column-wise 인덱싱: input[global_batch_idx, feature_idx]
        thread_max = rebind[Scalar[dtype]](input[global_batch_idx, feature_idx])

    shared_max[local_i] = thread_max
    barrier()

    # 트리 기반 병렬 리덕션으로 블록 내 최댓값 찾기
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

    var block_max = shared_max[0]  # 현재 블록의 최댓값

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-2단계: 지수 계산 및 블록별 합계 구하기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var exp_val: Scalar[dtype] = 0.0
    if global_batch_idx < batch_size:
        exp_val = rebind[Scalar[dtype]](
            exp(input[global_batch_idx, feature_idx] - block_max)
        )

    shared_sum[local_i] = exp_val
    barrier()

    # 트리 기반 병렬 리덕션으로 블록 내 지수 합계 구하기
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

    var block_sum = shared_sum[0]  # 현재 블록의 지수 합계

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1-3단계: 블록별 결과를 중간 버퍼에 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if local_i == 0:  # 블록의 대표 스레드만 저장
        block_maxes[block_id] = rebind[Scalar[dtype]](block_max)
        block_sums[block_id] = rebind[Scalar[dtype]](block_sum)


fn reduce_interim_kernel_2d_colwise[
    dtype: DType = DType.float32,
](
    final_vals: UnsafePointer[
        Scalar[dtype]
    ],  # [feature_size * 2] (각 열의 [global_max, global_sum])
    block_maxes: UnsafePointer[Scalar[dtype]],
    block_sums: UnsafePointer[Scalar[dtype]],
    feature_size: Int,
    blocks_per_col: Int,
):
    # ═══════════════════════════════════════════════════════════════════════════
    # Column-wise: 2단계 - 각 열별 글로벌 통계 계산
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 핵심: 각 열에 대해 독립적으로 글로벌 max와 sum 계산
    # - final_vals 구조: [col0_max, col0_sum, col1_max, col1_sum, ...]

    local_i = thread_idx.x
    feature_idx = block_idx.x  # 각 블록이 하나의 특성(열)을 담당

    # 경계 체크
    if feature_idx >= feature_size:
        return

    # 공유 메모리 할당
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-1단계: 현재 열의 글로벌 최댓값 찾기
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var thread_max: Scalar[dtype] = min_finite[dtype]()
    if local_i < blocks_per_col:
        # 현재 열의 block_idx 계산
        col_block_idx = feature_idx * blocks_per_col + local_i
        thread_max = block_maxes[col_block_idx]

    shared_max[local_i] = thread_max
    barrier()

    # 트리 리덕션으로 열별 글로벌 최댓값 찾기
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

    var global_max = shared_max[0]  # 현재 열의 글로벌 최댓값

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-2단계: 현재 열의 글로벌 합계 계산
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var adjusted_sum: Scalar[dtype] = 0.0
    if local_i < blocks_per_col:
        col_block_idx = feature_idx * blocks_per_col + local_i
        adjusted_sum = block_sums[col_block_idx] * rebind[Scalar[dtype]](
            exp(block_maxes[col_block_idx] - global_max)
        )

    shared_sum[local_i] = adjusted_sum
    barrier()

    # 트리 리덕션으로 열별 글로벌 합계 구하기
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

    var global_sum = shared_sum[0]  # 현재 열의 글로벌 합계

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-3단계: 열별 결과를 최종 버퍼에 저장
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if local_i == 0:
        final_vals[feature_idx * 2] = rebind[Scalar[dtype]](
            global_max
        )  # [feature_idx][0]
        final_vals[feature_idx * 2 + 1] = rebind[Scalar[dtype]](
            global_sum
        )  # [feature_idx][1]


fn normalize_kernel_2d_colwise[
    layout: Layout,
    batch_size: Int,
    feature_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
    final_vals: UnsafePointer[Scalar[dtype]],  # [feature_size * 2]
):
    # ═══════════════════════════════════════════════════════════════════════════
    # Column-wise: 3단계 - 각 열별 최종 정규화
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # 핵심: 각 열에 대해 독립적으로 최종 softmax 정규화 수행
    # output[batch_idx, feature_idx] = exp(input[batch_idx, feature_idx] - global_max) / global_sum

    local_i = thread_idx.x
    block_id = block_idx.x

    # Column-wise 인덱싱: 어떤 열(feature)과 어떤 블록(batch chunk)인지 계산
    blocks_per_col = (batch_size + TPB - 1) // TPB
    feature_idx = block_id // blocks_per_col
    batch_block_idx = block_id % blocks_per_col

    # 경계 체크
    if feature_idx >= feature_size:
        return

    # 현재 스레드가 담당할 전역 배치 인덱스
    global_batch_idx = batch_block_idx * TPB + local_i

    # 경계 체크: 유효한 배치 범위 내에서만 처리
    if global_batch_idx >= batch_size:
        return

    # 현재 열의 글로벌 통계 가져오기
    var global_max = final_vals[feature_idx * 2]  # 현재 열의 global_max
    var global_sum = final_vals[feature_idx * 2 + 1]  # 현재 열의 global_sum

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3단계: 최종 softmax 정규화
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공식: output[i,j] = exp(input[i,j] - global_max_j) / global_sum_j
    # - 각 열(j)별로 독립적인 global_max_j, global_sum_j 사용
    # - 스트라이드 메모리 접근 패턴 (캐시 효율성 다소 떨어짐)
    output[global_batch_idx, feature_idx] = rebind[Scalar[dtype]](
        exp(input[global_batch_idx, feature_idx] - global_max) / global_sum
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
        mode: StaticString,  # "row_wise" or "col_wise"
        batch_size: Int,
        feature_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=2],
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

            # 변수들을 미리 선언 (스코프 문제 해결)
            var blocks_per_row: Int = 0
            var blocks_per_col: Int = 0
            var total_blocks: Int

            @parameter
            if mode == "row_wise":
                # Row-wise: 2D Batched Softmax를 위한 그리드 계산
                blocks_per_row = (feature_size + TPB - 1) // TPB
                total_blocks = batch_size * blocks_per_row
            elif mode == "col_wise":
                # Column-wise: 2D Batched Softmax를 위한 그리드 계산
                blocks_per_col = (batch_size + TPB - 1) // TPB
                total_blocks = feature_size * blocks_per_col
            else:
                raise Error("Unsupported mode: " + mode)

            # 디바이스 버퍼 할당 (공통)
            var block_maxes_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                total_blocks
            )
            var block_sums_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                total_blocks
            )

            # Final values buffer 크기 계산
            var final_vals_size: Int = 0

            @parameter
            if mode == "row_wise":
                final_vals_size = (
                    batch_size * 2
                )  # 각 행의 [global_max, global_sum]
            elif mode == "col_wise":
                final_vals_size = (
                    feature_size * 2
                )  # 각 열의 [global_max, global_sum]

            var final_vals_buffer = gpu_ctx.enqueue_create_buffer[dtype](
                final_vals_size
            )

            # 2D Batched Softmax에서 GPU 메모리 초기화의 중요성
            # - 각 행별로 독립적인 softmax 계산
            # - 배치 차원만큼 증가한 중간 버퍼들의 정확한 초기화 필수
            #
            # 출력 텐서 초기화
            var total_elements = batch_size * feature_size
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    total_elements,
                    owning=False,
                ),
                0,
            )

            # 중간 버퍼들 초기화
            # - block_maxes_buffer: 각 행별 블록 최댓값 저장용
            # - block_sums_buffer: 각 행별 블록 합계 저장용 (반드시 0이어야 함)
            # - final_vals_buffer: 각 행별 최종 [global_max, global_sum] 저장용
            gpu_ctx.enqueue_memset(block_maxes_buffer, 0.0)
            gpu_ctx.enqueue_memset(block_sums_buffer, 0.0)
            gpu_ctx.enqueue_memset(final_vals_buffer, 0.0)

            # 2D Batched Softmax 3단계 파이프라인 실행 (mode에 따라 분기)

            @parameter
            if mode == "row_wise":
                # Row-wise: 각 행별 독립 처리

                # 1단계: 각 행별 블록 단위 리덕션 커널 실행
                gpu_ctx.enqueue_function[
                    reduce_block_kernel_2d[
                        layout, batch_size, feature_size, dtype
                    ]
                ](
                    block_maxes_buffer.unsafe_ptr(),
                    block_sums_buffer.unsafe_ptr(),
                    input_tensor,
                    grid_dim=(total_blocks, 1),
                    block_dim=THREADS_PER_BLOCK,
                )

                # 2단계: 각 행별 글로벌 통계 계산 커널 실행
                gpu_ctx.enqueue_function[reduce_interim_kernel_2d[dtype]](
                    final_vals_buffer.unsafe_ptr(),
                    block_maxes_buffer.unsafe_ptr(),
                    block_sums_buffer.unsafe_ptr(),
                    batch_size,
                    blocks_per_row,
                    grid_dim=(batch_size, 1),
                    block_dim=THREADS_PER_BLOCK,
                )

                # 3단계: 각 행별 최종 정규화 커널 실행
                gpu_ctx.enqueue_function[
                    normalize_kernel_2d[layout, batch_size, feature_size, dtype]
                ](
                    output_tensor,
                    input_tensor,
                    final_vals_buffer.unsafe_ptr(),
                    grid_dim=(total_blocks, 1),
                    block_dim=THREADS_PER_BLOCK,
                )

            elif mode == "col_wise":
                # Column-wise: 각 열별 독립 처리

                # 1단계: 각 열별 블록 단위 리덕션 커널 실행
                gpu_ctx.enqueue_function[
                    reduce_block_kernel_2d_colwise[
                        layout, batch_size, feature_size, dtype
                    ]
                ](
                    block_maxes_buffer.unsafe_ptr(),
                    block_sums_buffer.unsafe_ptr(),
                    input_tensor,
                    grid_dim=(total_blocks, 1),
                    block_dim=THREADS_PER_BLOCK,
                )

                # 2단계: 각 열별 글로벌 통계 계산 커널 실행
                gpu_ctx.enqueue_function[
                    reduce_interim_kernel_2d_colwise[dtype]
                ](
                    final_vals_buffer.unsafe_ptr(),
                    block_maxes_buffer.unsafe_ptr(),
                    block_sums_buffer.unsafe_ptr(),
                    feature_size,
                    blocks_per_col,
                    grid_dim=(feature_size, 1),
                    block_dim=THREADS_PER_BLOCK,
                )

                # 3단계: 각 열별 최종 정규화 커널 실행
                gpu_ctx.enqueue_function[
                    normalize_kernel_2d_colwise[
                        layout, batch_size, feature_size, dtype
                    ]
                ](
                    output_tensor,
                    input_tensor,
                    final_vals_buffer.unsafe_ptr(),
                    grid_dim=(total_blocks, 1),
                    block_dim=THREADS_PER_BLOCK,
                )

        elif target == "cpu":

            @parameter
            if mode == "row_wise":
                # CPU Row-wise: 각 행별 softmax 처리
                for batch_idx in range(batch_size):
                    # 행별 최댓값 찾기
                    var row_max: Scalar[dtype] = min_finite[dtype]()
                    for feature_idx in range(feature_size):
                        row_max = max(
                            row_max,
                            rebind[Scalar[dtype]](
                                input_tensor[batch_idx, feature_idx]
                            ),
                        )

                    # 행별 지수 합계 계산
                    var row_sum: Scalar[dtype] = 0.0
                    for feature_idx in range(feature_size):
                        var exp_val = rebind[Scalar[dtype]](
                            exp(input_tensor[batch_idx, feature_idx] - row_max)
                        )
                        output_tensor[batch_idx, feature_idx] = exp_val
                        row_sum += exp_val

                    # 행별 정규화
                    for feature_idx in range(feature_size):
                        output_tensor[batch_idx, feature_idx] = (
                            output_tensor[batch_idx, feature_idx] / row_sum
                        )

            elif mode == "col_wise":
                # CPU Column-wise: 각 열별 softmax 처리
                for feature_idx in range(feature_size):
                    # 열별 최댓값 찾기
                    var col_max: Scalar[dtype] = min_finite[dtype]()
                    for batch_idx in range(batch_size):
                        col_max = max(
                            col_max,
                            rebind[Scalar[dtype]](
                                input_tensor[batch_idx, feature_idx]
                            ),
                        )

                    # 열별 지수 합계 계산
                    var col_sum: Scalar[dtype] = 0.0
                    for batch_idx in range(batch_size):
                        var exp_val = rebind[Scalar[dtype]](
                            exp(input_tensor[batch_idx, feature_idx] - col_max)
                        )
                        output_tensor[batch_idx, feature_idx] = exp_val
                        col_sum += exp_val

                    # 열별 정규화
                    for batch_idx in range(batch_size):
                        output_tensor[batch_idx, feature_idx] = (
                            output_tensor[batch_idx, feature_idx] / col_sum
                        )
        else:
            raise Error("Unsupported target: " + target)
