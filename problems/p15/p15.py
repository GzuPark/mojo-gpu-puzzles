from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray

# MAX Graph와 Python-Mojo 통합 예제
#
# Python-GPU 브리지
# 이전 퍼즐들에서는 순수 Mojo로 GPU 커널을 작성했지만,
# 이제는 Python에서 Mojo GPU 커널을 호출하는 방법을 학습합니다.
#
# 📊 MAX Graph 시스템의 핵심 구성 요소:
# 1. Graph: 계산 그래프 정의 (연산들의 연결 관계)
# 2. TensorType: 입력/출력 텐서의 타입과 형태 정의
# 3. ops.custom: 사용자 정의 Mojo 연산 호출
# 4. InferenceSession: 그래프 컴파일 및 실행 관리
# 5. DeviceRef: GPU/CPU 디바이스 추상화

def conv_1d(
    input: NDArray[np.float32],
    kernel: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
) -> Tensor:
    """
    1D 컨볼루션을 위한 MAX Graph 래퍼 함수

    선언적 그래프 정의
    이전에는 명령형으로 GPU 커널을 직접 호출했지만,
    MAX Graph에서는 계산 그래프를 먼저 정의하고 나중에 실행합니다.

    Args:
        input: NumPy 입력 배열 (Python 생태계 호환성)
        kernel: 컨볼루션 커널 (NumPy 배열)
        session: MAX 추론 세션 (그래프 컴파일/실행 관리)
        device: 실행 디바이스 (CPU/GPU 추상화)

    Returns:
        Tensor: MAX 드라이버 텐서 (GPU 메모리에 저장된 결과)
    """
    dtype = DType.float32

    # NumPy → MAX Tensor 변환
    # 이전: Mojo 내에서 직접 메모리 버퍼 생성
    # 현재: Python NumPy 배열을 MAX Tensor로 변환 후 디바이스로 이동
    #
    # Tensor.from_numpy(): NumPy 배열을 MAX Tensor로 변환
    # .to(device): 텐서를 지정된 디바이스(GPU/CPU)로 이동
    # 이는 PyTorch의 .to(device) 패턴과 유사합니다.
    input_tensor = Tensor.from_numpy(input).to(device)
    kernel_tensor = Tensor.from_numpy(kernel).to(device)

    # 사용자 정의 연산 패키지 로딩
    # 이전: 같은 파일 내에서 함수 직접 호출
    # 현재: 별도 디렉토리의 Mojo 패키지를 동적 로딩
    mojo_kernels = Path(__file__).parent / "op"

    # 계산 그래프 정의 (선언적 프로그래밍)
    # 이전: 명령형 스타일 - 커널을 즉시 실행
    # 현재: 선언형 스타일 - 그래프를 먼저 정의, 나중에 실행
    #
    # Graph 생성 매개변수:
    # - name: 그래프 식별자 ("conv_1d_graph")
    # - input_types: 입력 텐서들의 타입 명세 (컴파일 타임 최적화용)
    # - custom_extensions: 사용자 정의 Mojo 연산 패키지 경로
    with Graph(
        "conv_1d_graph",
        input_types=[
            # TensorType - 타입 안전성과 최적화
            # 이전: 런타임에 텐서 형태 확인
            # 현재: 컴파일 타임에 텐서 타입/형태/디바이스 명세
            #
            # TensorType 구성 요소:
            # - dtype: 데이터 타입 (DType.float32)
            # - shape: 텐서 형태 (input_tensor.shape)
            # - device: 실행 디바이스 (DeviceRef.from_device)
            TensorType(
                dtype,
                shape=input_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=kernel_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        # 동적 확장 로딩
        # MAX Graph가 런타임에 Mojo 패키지를 로드하여
        # 사용자 정의 연산을 사용할 수 있게 합니다.
        custom_extensions=[mojo_kernels],
    ) as graph:
        # 그래프 입력 노드 정의
        # 이전: 함수 매개변수로 직접 전달
        # 현재: 그래프의 입력 노드로 추상화
        #
        # graph.inputs: 위에서 정의한 input_types 순서대로 반환
        # input_value, kernel_value: 그래프 내에서 사용할 심볼릭 값들
        input_value, kernel_value = graph.inputs

        # 사용자 정의 연산 호출 (ops.custom)
        # 이전: ctx.enqueue_function으로 직접 GPU 커널 호출
        # 현재: ops.custom으로 Mojo 연산을 그래프 노드로 추가
        #
        # ops.custom 매개변수 설명:
        # - name: Mojo에서 @compiler.register로 등록한 연산 이름
        # - values: 입력 값들 (그래프 노드들)
        # - out_types: 출력 텐서 타입 명세
        # - parameters: 컴파일 타임 매개변수 (Mojo 함수의 [] 매개변수에 해당)
        output = ops.custom(
            name="conv1d",  # conv1d.mojo의 @compiler.register("conv1d")와 매칭
            device=DeviceRef.from_device(device),
            values=[input_value, kernel_value],
            out_types=[
                # 출력 텐서 타입: 입력과 동일한 형태/타입/디바이스
                TensorType(
                    dtype=input_value.tensor.dtype,
                    shape=input_value.tensor.shape,  # 1D conv에서 출력 크기 = 입력 크기
                    device=DeviceRef.from_device(device),
                )
            ],
            # 컴파일 타임 매개변수 전달
            # 이전: Mojo 함수 호출 시 [] 안에 직접 전달
            # 현재: parameters 딕셔너리로 전달
            #
            # 이 매개변수들은 Mojo의 Conv1DCustomOp.execute 함수의
            # [input_size: Int, conv_size: Int, dtype: DType] 매개변수에 대응됩니다.
            parameters={
                "input_size": input_tensor.shape[0],
                "conv_size": kernel_tensor.shape[0],
                "dtype": dtype,
            },
        )[0].tensor  # ops.custom은 리스트를 반환하므로 첫 번째 결과 선택

        # 그래프 출력 정의
        # 이전: 함수 반환값으로 직접 반환
        # 현재: graph.output()으로 그래프의 출력 노드 명시
        graph.output(output)

    # 그래프 컴파일 (지연 실행)
    # 이전: 커널 함수가 즉시 실행됨
    # 현재: 그래프를 먼저 컴파일하여 최적화된 실행 계획 생성
    #
    # session.load(graph):
    # 1. 그래프를 디바이스별 최적화된 코드로 컴파일
    # 2. 메모리 할당 계획 수립
    # 3. 실행 가능한 모델 객체 반환
    print("Compiling 1D convolution graph...")
    model = session.load(graph)

    # 컴파일된 모델 실행
    # 이전: 커널 실행과 동시에 결과 생성
    # 현재: 미리 컴파일된 모델에 실제 데이터 전달하여 실행
    #
    # model.execute():
    # 1. 입력 텐서들을 그래프에 바인딩
    # 2. 최적화된 실행 계획에 따라 연산 수행
    # 3. 결과 텐서 반환
    print("Executing 1D convolution...")
    result = model.execute(input_tensor, kernel_tensor)[0]

    # 디바이스 간 데이터 이동
    # 이전: GPU 메모리에서 직접 결과 확인
    # 현재: GPU 결과를 CPU로 복사하여 Python에서 접근 가능하게 함
    #
    # result.to(CPU()): GPU 메모리의 결과를 CPU 메모리로 복사
    # 이는 PyTorch의 .cpu() 메서드와 유사한 개념입니다.
    assert isinstance(result, Tensor)
    return result.to(CPU())


if __name__ == "__main__":
    # 하이브리드 Python-GPU 프로그래밍 설정
    INPUT_SIZE = 15
    KERNEL_SIZE = 4

    # 디바이스 추상화 및 자동 선택
    # 이전: DeviceContext()로 GPU 강제 사용
    # 현재: accelerator_count()로 사용 가능한 GPU 확인 후 자동 선택
    #
    # accelerator_count(): 시스템의 사용 가능한 GPU 개수 반환
    # GPU가 없으면 CPU 폴백, 있으면 GPU 사용 (이식성 향상)
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # 추론 세션 관리
    # 이전: DeviceContext로 단순한 GPU 컨텍스트 관리
    # 현재: InferenceSession으로 복잡한 ML 워크플로우 관리
    #
    # InferenceSession 기능:
    # 1. 다중 디바이스 관리 (devices=[device])
    # 2. 그래프 컴파일 캐싱
    # 3. 메모리 풀 관리
    # 4. 실행 최적화
    session = InferenceSession(devices=[device])

    # Python 생태계 통합
    # 이전: Mojo 배열로 직접 데이터 생성
    # 현재: NumPy 배열 사용으로 Python ML 생태계와 호환
    #
    # np.arange(): NumPy의 표준 배열 생성 함수
    # dtype=np.float32: NumPy 타입을 명시적으로 지정
    # 이는 pandas, scikit-learn 등과 호환됩니다.
    input_array = np.arange(INPUT_SIZE, dtype=np.float32)
    kernel = np.arange(KERNEL_SIZE, dtype=np.float32)

    # 검증용 참조 구현 (NumPy 버전)
    # 이전: CPU에서 단순한 반복문으로 검증
    # 현재: NumPy의 벡터화 연산과 유사한 패턴으로 검증
    #
    # 이는 실제 ML 개발에서 사용하는 패턴입니다:
    # 1. NumPy로 프로토타입 구현
    # 2. 사용자 정의 GPU 커널로 가속화
    # 3. 결과 비교로 정확성 검증
    expected_result = np.zeros_like(input_array, dtype=np.float32)
    for i in range(INPUT_SIZE):
        for j in range(KERNEL_SIZE):
            if i + j < INPUT_SIZE:
                expected_result[i] += input_array[i + j] * kernel[j]

    print(f"Input array: {input_array}")
    print(f"Convolution kernel: {kernel}")
    print(f"Expected result (NumPy calculation): {expected_result}")

    # Python-Mojo 통합 실행
    # 이전: 순수 Mojo 환경에서 실행
    # 현재: Python에서 Mojo GPU 커널 호출
    result = conv_1d(input_array, kernel, session, device)
    print(f"1D Convolution result (custom Mojo kernel): {result.to_numpy()}")

    # 크로스 플랫폼 결과 검증
    # 이전: assert_equal로 단순 비교
    # 현재: np.testing.assert_allclose로 부동소수점 오차 허용
    #
    # rtol=1e-5: 상대 오차 허용 범위 (GPU/CPU 간 미세한 계산 차이 고려)
    # 이는 실제 ML 개발에서 필수적인 패턴입니다.
    np.testing.assert_allclose(result.to_numpy(), expected_result, rtol=1e-5)
    print("Verification passed: Custom kernel results match NumPy calculation")

# 1. **선언적 그래프 프로그래밍**: 계산을 먼저 정의하고 나중에 실행
# 2. **Python-Mojo 브리지**: NumPy ↔ MAX Tensor ↔ Mojo LayoutTensor
# 3. **타입 안전성**: TensorType으로 컴파일 타임 타입 검증
# 4. **디바이스 추상화**: CPU/GPU를 통일된 인터페이스로 관리
# 5. **패키지 시스템**: Mojo 코드를 재사용 가능한 패키지로 배포
# 6. **ML 생태계 통합**: NumPy, PyTorch 등과 호환되는 워크플로우
