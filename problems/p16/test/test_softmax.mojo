from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_almost_equal

from op import softmax_gpu_kernel, softmax_cpu_kernel

# 확률 분포 테스트 (Probability Distribution Testing)
#
# Softmax 테스트에서 확인해야 할 사항:
# 1. 모든 출력값이 0 이상 (음수 확률 없음)
# 2. 모든 출력값의 합이 1.0 (정규화 확인)
# 3. CPU와 GPU 결과의 일치성 (구현 정확성)
# 4. 수치적 안정성 (큰 입력값에서도 안정적 동작)

alias SIZE = 128
alias TPB = 128
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (TPB, 1)
alias layout = Layout.row_major(SIZE)
alias dtype = DType.float32


def test_softmax():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[DType.float32](SIZE).enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[DType.float32](SIZE).enqueue_fill(0)

        # for CPU testing
        expected = ctx.enqueue_create_host_buffer[DType.float32](
            SIZE
        ).enqueue_fill(0)
        expected_tensor = LayoutTensor[mut=True, dtype, layout](
            expected.unsafe_ptr()
        )

        # 순차적 증가 값으로 실제 신경망 출력과 유사한 패턴 생성
        #
        # 왜 순차적 데이터를 사용하는가?
        # 1. 예측 가능한 결과로 디버깅 용이
        # 2. 큰 값에서 작은 값까지 다양한 범위 테스트
        # 3. Softmax의 수치적 안정성 확인 (큰 값들이 포함됨)
        # Initialize input with more reasonable values
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                inp_host[i] = Float32(i)

            print("Input values:")
            for i in range(SIZE):
                print(inp_host[i], end=" ")
            print()
            # Create layout tensors for CPU calculation
            input_host_tensor = LayoutTensor[mut=True, dtype, layout](
                inp_host.unsafe_ptr()
            )

        # for GPU testing
        output_tensor = LayoutTensor[mut=True, dtype, layout](out.unsafe_ptr())
        input_tensor = LayoutTensor[mut=True, dtype, layout](inp.unsafe_ptr())

        # CPU를 기준값으로 사용하는 검증 패턴
        # 1. CPU 구현이 더 간단하고 이해하기 쉬워서 버그 가능성이 낮음
        # 2. GPU 병렬 처리의 복잡성을 순차 처리로 검증 가능
        # 3. 동일한 알고리즘의 서로 다른 구현 간 일치성 확인
        # Compute expected results using our CPU kernel
        softmax_cpu_kernel[layout, SIZE, dtype](
            expected_tensor, input_host_tensor
        )

        # Run GPU kernel
        ctx.enqueue_function[softmax_gpu_kernel[layout, SIZE, dtype]](
            output_tensor,
            input_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("GPU softmax results:")
            for i in range(SIZE):
                print(out_host[i], end=" ")
            print()

            print("Expected results:")
            for i in range(SIZE):
                print(expected[i], end=" ")
            print()

            # 확률 분포 유효성 검증
            # 1. 개별 값의 정확성 (assert_almost_equal)
            # 2. 전체 확률의 합이 1.0인지 확인
            # 3. 부동소수점 오차 허용 (atol, rtol 설정)
            var sum_gpu: Float32 = 0.0
            for i in range(SIZE):
                sum_gpu += out_host[i]
                assert_almost_equal(
                    out_host[i], expected[i], atol=1e-5, rtol=1e-5
                )

            print("Sum of probabilities:", sum_gpu)

            # 확률 분포 정규화 검증
            # - Softmax 구현이 올바른지 확인하는 핵심 테스트
            assert_almost_equal(sum_gpu, 1.0, atol=1e-5, rtol=1e-5)
            print("All tests passed 🎉")
