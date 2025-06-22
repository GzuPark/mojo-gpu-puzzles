from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from utils.numerics import max_finite, min_finite

# Softmax GPU 커널 구현
#
# 수치적 안정성 (Numerical Stability)
# - 지수 함수 사용으로 인한 오버플로우/언더플로우 위험
#
# Softmax의 수치적 문제:
# 1. exp(x)는 x가 클 때 무한대로 발산 (오버플로우)
# 2. exp(x)는 x가 작을 때 0으로 수렴 (언더플로우)
# 3. 해결책: exp(x - max(x)) 사용 → 최대값이 exp(0) = 1이 됨

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

    # 병렬 리덕션을 위한 공유 메모리 (Parallel Reduction with Shared Memory)
    # - 모든 스레드가 협력하여 최댓값과 합계를 계산
    #
    # 공유 메모리 사용 이유:
    # 1. 모든 스레드가 동일한 최댓값에 접근해야 함
    # 2. 모든 스레드가 동일한 합계값에 접근해야 함
    # 3. 글로벌 메모리보다 100배 이상 빠른 접근 속도
    shared_max = tb[dtype]().row_major[TPB]().shared().alloc()
    shared_sum = tb[dtype]().row_major[TPB]().shared().alloc()

    var thread_max: Scalar[dtype] = min_finite[dtype]()

    if global_i < input_size:
        thread_max = rebind[Scalar[dtype]](input[global_i])

    shared_max[local_i] = thread_max

    barrier()

    # 트리 기반 병렬 리덕션 (Tree-based Parallel Reduction)
    # - 병렬 트리 리덕션 O(log n)
    #
    # 동작 원리:
    # 1단계: [a,b,c,d,e,f,g,h] → [max(a,e), max(b,f), max(c,g), max(d,h)]
    # 2단계: [max(a,e), max(b,f), max(c,g), max(d,h)] → [max(max(a,e),max(c,g)), max(max(b,f),max(d,h))]
    # 3단계: [max(...), max(...)] → [global_max]
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

    block_max = shared_max[0]

    # 수치적 안정성을 위한 최댓값 차감
    # - exp(x - max) 계산으로 오버플로우 방지
    #
    # 수학적 정당성:
    # softmax(x_i) = exp(x_i) / Σexp(x_j)
    #              = exp(x_i - c) / Σexp(x_j - c)  (c는 임의의 상수)
    # c = max(x)로 선택하면 모든 지수가 0 이하가 되어 안전함
    var exp_val: Scalar[dtype] = 0.0

    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(input[global_i] - block_max))
        output[global_i] = exp_val

    shared_sum[local_i] = exp_val

    barrier()

    # 합계를 위한 두 번째 병렬 리덕션
    # - 최댓값 구한 후 정규화를 위한 합계도 구해야 함
    #
    # Softmax 특성상 두 단계 리덕션 필요:
    # 1단계: 최댓값 구하기 (수치 안정성)
    # 2단계: 지수의 합 구하기 (정규화)
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

    block_sum = shared_sum[0]

    # 새로운 개념: 확률 분포로의 정규화
    # 이전: 계산 결과를 그대로 출력
    # 현재: 합으로 나누어 확률 분포 생성 (모든 값의 합 = 1.0)
    #
    # 정규화 공식: softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)
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
    # GPU vs CPU 구현 차이점:
    # - GPU: 병렬 리덕션으로 O(log n) 시간 복잡도
    # - CPU: 순차 반복문으로 O(n) 시간 복잡도
    # - GPU: 공유 메모리 사용
    # - CPU: 지역 변수 사용
    # FILL IN (roughly 10 lines)

    # 1단계: 최댓값 찾기 (수치 안정성)
    var max_val: Scalar[dtype] = min_finite[dtype]()

    for i in range(input_size):
        max_val = max(max_val, rebind[Scalar[dtype]](input[i]))

    # 2단계: 지수 계산 및 합계 구하기
    var sum_exp: Scalar[dtype] = 0.0

    for i in range(input_size):
        var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
        output[i] = exp_val
        sum_exp += exp_val

    # 3단계: 정규화 (확률 분포 생성)
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
