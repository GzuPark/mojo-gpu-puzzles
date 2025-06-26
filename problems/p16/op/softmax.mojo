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

alias SIZE = 128
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)


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

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=(TPB, 1),
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
