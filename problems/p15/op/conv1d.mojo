from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb

# Custom Operation Package
#
# 이전 퍼즐과의 차이점:
# - 이전: 순수 Mojo 환경에서 GPU 커널 작성 및 실행
# - 현재: Python에서 호출 가능한 Mojo 패키지 생성
# - 이전: main() 함수에서 직접 커널 테스트
# - 현재: @compiler.register 데코레이터로 연산 등록
#
# 📦 패키지 구조의 이해:
# problems/p15/op/
# ├── __init__.mojo          # 패키지 초기화 파일
# └── conv1d.mojo           # 이 파일 - 사용자 정의 연산 구현
#
# 이 구조는 Python의 패키지 시스템과 유사하며,
# MAX Graph에서 동적으로 로드할 수 있는 Mojo 모듈을 제공합니다.

# ANCHOR: conv1d_kernel
alias TPB = 15
alias BLOCKS_PER_GRID = (2, 1)


# 기존 GPU 커널 재사용
# 이 커널은 Puzzle 11에서 학습한 것과 동일한 로직입니다.
# 새로운 점은 이제 이 커널이 Python에서 호출 가능한 패키지의 일부라는 것입니다.
#
# 커널 기능 요약 (Puzzle 11에서 학습한 내용):
# - 1D 컨볼루션 연산 수행
# - 공유 메모리를 사용한 성능 최적화
# - 블록 경계 데이터 처리
# - 제로 패딩을 통한 경계 조건 처리
fn conv1d_kernel[
    in_layout: Layout,
    out_layout: Layout,
    conv_layout: Layout,
    input_size: Int,
    conv_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    input: LayoutTensor[mut=True, dtype, in_layout],
    kernel: LayoutTensor[mut=True, dtype, conv_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # first: need to account for padding
    shared_a = tb[dtype]().row_major[TPB + conv_size - 1]().shared().alloc()
    shared_b = tb[dtype]().row_major[conv_size]().shared().alloc()
    if global_i < input_size:
        shared_a[local_i] = input[global_i]

    # second: load elements needed for convolution at block boundary
    if local_i < conv_size - 1:
        # indices from next block
        next_idx = global_i + TPB
        if next_idx < input_size:
            shared_a[TPB + local_i] = input[next_idx]

    if local_i < conv_size:
        shared_b[local_i] = kernel[local_i]

    barrier()

    if global_i < input_size:
        var local_sum: output.element_type = 0

        @parameter
        for j in range(conv_size):
            if local_i + j < TPB + conv_size - 1:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum


# ANCHOR_END: conv1d_kernel


# ANCHOR: conv1d_custom_op
import compiler  # @compiler.register 데코레이터 제공
from runtime.asyncrt import DeviceContextPtr  # 비동기 런타임 컨텍스트
from tensor import InputTensor, OutputTensor  # MAX Graph 텐서 인터페이스
from memory import UnsafePointer
from gpu.host import DeviceBuffer  # GPU 메모리 버퍼 관리


# 사용자 정의 연산 등록 (@compiler.register)
#
# @compiler.register 데코레이터의 역할:
# 1. Mojo 함수를 MAX Graph에서 호출 가능한 연산으로 등록
# 2. Python의 ops.custom(name="conv1d", ...)에서 이 이름으로 호출
# 3. 컴파일 타임에 연산 메타데이터 생성
# 4. 런타임에 동적 디스패치 가능
#
# 이는 PyTorch의 custom C++ 연산이나 TensorFlow의 custom op와 유사한 개념입니다.
@compiler.register("conv1d")
struct Conv1DCustomOp:
    # 정적 메서드를 통한 연산 인터페이스
    # 이전: 일반 함수로 GPU 커널 정의
    # 현재: struct 내의 정적 메서드로 MAX Graph 인터페이스 제공
    #
    # 왜 struct와 정적 메서드를 사용하는가?
    # 1. 네임스페이스 격리 (Conv1DCustomOp.execute)
    # 2. 메타데이터 첨부 가능 (@compiler.register와 연결)
    # 3. 타입 안전성 향상
    # 4. 여러 연산을 하나의 패키지에 포함 가능
    @staticmethod
    fn execute[
        # 컴파일 타임 매개변수
        # 이전: 런타임에 값 전달
        # 현재: 컴파일 타임에 최적화를 위한 매개변수 전달
        #
        # target: "cpu" 또는 "gpu" - 실행 대상 디바이스
        # input_size, conv_size: Python에서 전달된 크기 정보
        # dtype: 데이터 타입 (컴파일 타임 최적화용)
        target: StaticString,  # 컴파일 타임 문자열 상수
        input_size: Int,
        conv_size: Int,
        dtype: DType = DType.float32,
    ](
        # MAX Graph 텐서 인터페이스
        # 이전: LayoutTensor 직접 사용
        # 현재: InputTensor/OutputTensor 추상화 레이어
        #
        # InputTensor/OutputTensor의 장점:
        # 1. Python-Mojo 간 타입 안전성
        # 2. 자동 메모리 관리
        # 3. 디바이스 간 투명한 데이터 이동
        # 4. MAX Graph 최적화 엔진과 통합
        output: OutputTensor[rank=1],  # 1차원 출력 텐서
        input: InputTensor[rank=1],  # 입력 텐서 (1차원)
        kernel: InputTensor[rank=1],  # 커널 텐서 (1차원)
        # Device Context Pointer
        # 이전: DeviceContext를 직접 생성하여 사용
        # 현재: MAX Graph에서 관리하는 컨텍스트를 포인터로 전달받음
        #
        # DeviceContextPtr의 역할:
        # 1. GPU 메모리 할당/해제 관리
        # 2. 커널 실행 스케줄링
        # 3. 디바이스 간 데이터 동기화
        # 4. 에러 처리 및 복구
        ctx: DeviceContextPtr,
    ) raises:
        # MAX Graph 텐서 → LayoutTensor 변환
        # 이전: 직접 LayoutTensor 사용
        # 현재: MAX Graph 텐서를 LayoutTensor로 변환하여 기존 커널 재사용
        #
        # .to_layout_tensor()의 역할:
        # 1. MAX Graph의 추상화된 텐서를 Mojo 네이티브 형식으로 변환
        # 2. 메모리 레이아웃 정보 추출
        # 3. GPU 포인터 접근 허용
        # 4. 기존 GPU 커널과의 호환성 제공
        output_tensor = output.to_layout_tensor()
        input_tensor = input.to_layout_tensor()
        kernel_tensor = kernel.to_layout_tensor()

        # 컴파일 타임 레이아웃 추출
        # 이전: 함수 매개변수로 레이아웃 전달
        # 현재: 런타임 텐서에서 컴파일 타임 레이아웃 정보 추출
        #
        # alias를 사용하는 이유:
        # 1. 컴파일 타임 상수로 처리 (성능 최적화)
        # 2. 타입 검증 강화
        # 3. GPU 커널의 매개변수 요구사항 충족
        alias in_layout = input_tensor.layout
        alias output_layout = output_tensor.layout
        alias conv_layout = kernel_tensor.layout

        # 조건부 컴파일 (@parameter if)
        # 이전: 런타임 조건문 사용
        # 현재: 컴파일 타임 조건문으로 디바이스별 최적화
        #
        # @parameter if의 장점:
        # 1. 사용하지 않는 코드 경로 제거 (코드 크기 감소)
        # 2. 디바이스별 특화 최적화 가능
        # 3. 타입 검증 강화
        # 4. 실행 시 분기 오버헤드 제거
        @parameter
        if target == "gpu":
            # 디바이스 컨텍스트 추출 및 메모리 관리
            # 이전: with DeviceContext() as ctx: 패턴 사용
            # 현재: MAX Graph에서 전달받은 컨텍스트 포인터 활용
            #
            # ctx.get_device_context()의 역할:
            # 1. 포인터에서 실제 DeviceContext 객체 추출
            # 2. GPU 리소스에 대한 접근 권한 획득
            # 3. 메모리 풀 및 스트림 관리 활성화
            gpu_ctx = ctx.get_device_context()

            # 명시적 메모리 초기화
            # 이전: .enqueue_fill(0)으로 버퍼 초기화
            # 현재: enqueue_memset으로 저수준 메모리 초기화
            #
            # 왜 명시적 초기화가 필요한가?
            # 1. MAX Graph에서 전달받은 출력 버퍼는 초기화되지 않을 수 있음
            # 2. 컨볼루션 연산에서 누적 합계를 올바르게 계산하기 위해 필요
            # 3. 메모리 안전성 보장
            # 4. 재현 가능한 결과 보장
            gpu_ctx.enqueue_memset(
                # DeviceBuffer를 통한 안전한 메모리 접근
                # 이전: unsafe_ptr() 직접 사용
                # 현재: DeviceBuffer로 래핑하여 타입 안전성 제공
                #
                # DeviceBuffer 구성 요소:
                # - output.dtype: 버퍼의 데이터 타입
                # - gpu_ctx: 메모리를 관리하는 디바이스 컨텍스트
                # - rebind[...]: 포인터 타입 변환 (타입 안전성 유지)
                # - input_size: 버퍼 크기 (요소 개수)
                # - owning=False: 메모리 소유권을 갖지 않음 (MAX Graph가 관리)
                DeviceBuffer[dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[dtype]]](output_tensor.ptr),
                    input_size,
                    owning=False,
                ),
                0,  # 초기화 값 (0으로 설정)
            )

            # 기존 GPU 커널의 재사용
            # 이전: 커널을 직접 호출
            # 현재: MAX Graph 컨텍스트 내에서 기존 커널 재사용
            #
            # enqueue_function의 역할:
            # 1. GPU 커널을 비동기 실행 큐에 추가
            # 2. 컴파일 타임 매개변수 전달 ([...] 부분)
            # 3. 런타임 인자 전달 (텐서들)
            # 4. 그리드/블록 차원 설정
            #
            # 이 호출이 Puzzle 11과 다른 점:
            # - DeviceContext가 MAX Graph에서 관리됨
            # - 텐서가 MAX Graph에서 제공됨
            # - 메모리 생명주기가 MAX Graph에서 관리됨
            gpu_ctx.enqueue_function[
                conv1d_kernel[
                    in_layout,
                    output_layout,
                    conv_layout,
                    input_size,
                    conv_size,
                ]
            ](
                output_tensor,
                input_tensor,
                kernel_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=(TPB, 1),
            )

        elif target == "cpu":
            # CPU 폴백 구현
            # 이전: GPU만 지원
            # 현재: CPU/GPU 하이브리드 지원
            #
            # CPU 폴백의 중요성:
            # 1. 개발 환경에서 GPU가 없을 때 테스트 가능
            # 2. 작은 데이터에서는 CPU가 더 효율적일 수 있음
            # 3. 디버깅 시 CPU에서 단계별 실행 가능
            # 4. 프로덕션 환경의 가용성 향상
            #
            # 현재는 구현되지 않았지만, 여기에 CPU 버전의 컨볼루션을 구현할 수 있습니다.
            # 예: NumPy 스타일의 순차 처리 또는 SIMD 최적화된 CPU 코드
            pass
        else:
            # 런타임 에러 처리
            # 이전: 컴파일 타임에 모든 경로가 결정됨
            # 현재: 런타임에 잘못된 대상이 전달될 수 있으므로 에러 처리 필요
            #
            # raises 키워드의 중요성:
            # 1. 함수가 에러를 발생시킬 수 있음을 명시
            # 2. 호출자에게 에러 처리 책임 전가
            # 3. MAX Graph 시스템의 에러 복구 메커니즘과 통합
            raise Error("Unsupported target: " + target)


# ANCHOR_END: conv1d_custom_op

# 1. **사용자 정의 연산 등록**: @compiler.register로 Mojo 함수를 MAX Graph에 노출
# 2. **Python-Mojo 브리지**: InputTensor/OutputTensor를 통한 안전한 데이터 교환
# 3. **컴파일 타임 최적화**: @parameter if로 디바이스별 코드 생성
# 4. **메모리 관리**: DeviceBuffer와 명시적 초기화를 통한 안전성 보장
# 5. **에러 처리**: raises 키워드를 통한 런타임 에러 전파
# 6. **코드 재사용**: 기존 GPU 커널을 MAX Graph 환경에서 재활용
#
# 🔗 전체 워크플로우:
# Python (p15.py) → MAX Graph → @compiler.register → Conv1DCustomOp.execute → conv1d_kernel
#
# 이 구조는 실제 프로덕션 ML 시스템에서 사용하는 패턴으로:
# - 연구자는 Python에서 편리하게 개발
# - 엔지니어는 Mojo로 고성능 커널 구현
# - MAX Graph가 둘 사이의 브리지 역할
# - 최종 사용자는 성능과 편의성을 모두 확보
