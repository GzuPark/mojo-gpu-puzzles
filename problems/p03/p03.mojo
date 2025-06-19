from memory import UnsafePointer
from gpu import thread_idx  # GPU 스레드의 인덱스 정보를 가져오기 위한 라이브러리
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from testing import assert_equal

# ANCHOR: add_10_guard
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias SIZE = 4  # 벡터의 크기 (4개의 요소)
alias BLOCKS_PER_GRID = 1  # GPU 그리드당 블록 수 (1개의 블록만 사용)
alias THREADS_PER_BLOCK = (8, 1)  # 블록당 스레드 수 (8개) - 데이터 크기보다 많음
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)


# GPU 커널 함수: 범위 검사가 포함된 각 요소에 10을 더하는 함수
# 이 함수는 스레드 수가 데이터 크기보다 클 때 안전하게 처리합니다
fn add_10_guard(
    output: UnsafePointer[Scalar[dtype]],  # 출력 결과를 저장할 메모리 포인터
    a: UnsafePointer[Scalar[dtype]],  # 입력 데이터가 저장된 메모리 포인터
    size: Int,  # 실제 데이터의 크기 (범위 검사용)
):
    # thread_idx.x: 현재 스레드의 x축 인덱스 (0~7 중 하나, 총 8개 스레드)
    # 하지만 실제 데이터는 4개만 있으므로 일부 스레드는 유효하지 않은 인덱스를 가짐
    i = thread_idx.x

    # FILL ME IN (roughly 2 lines)
    # 범위 검사: 스레드 인덱스가 실제 데이터 크기보다 작은지 확인
    # 이 검사가 없으면 스레드 4, 5, 6, 7은 유효하지 않은 메모리에 접근할 수 있음
    if i < size:  # i가 0, 1, 2, 3일 때만 실행 (4, 5, 6, 7은 건너뜀)
        # 유효한 인덱스일 때만 계산을 수행합니다
        output[i] = a[i] + 10.0


# ANCHOR_END: add_10_guard


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # 이 컨텍스트는 GPU 메모리 할당, 데이터 복사, 커널 실행 등을 담당합니다
    with DeviceContext() as ctx:
        # GPU 메모리 버퍼들을 생성하고 0으로 초기화합니다 (메서드 체이닝 사용)
        # 출력 버퍼와 입력 버퍼를 각각 SIZE(4) 크기로 생성합니다
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 출력 결과 버퍼
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 입력 버퍼

        # 입력 데이터를 초기화하기 위해 GPU 메모리를 호스트(CPU) 메모리로 매핑합니다
        with a.map_to_host() as a_host:
            # 입력 배열에 0, 1, 2, 3 값을 저장합니다
            for i in range(SIZE):
                a_host[i] = i  # 배열: [0, 1, 2, 3]

        # GPU 커널 함수를 실행합니다
        # 주목: 8개의 스레드를 사용하지만 데이터는 4개만 있습니다!
        # 이는 의도적으로 스레드 수 > 데이터 크기 상황을 시연하기 위함입니다
        ctx.enqueue_function[add_10_guard](
            out.unsafe_ptr(),  # 출력 버퍼의 메모리 포인터
            a.unsafe_ptr(),  # 입력 버퍼의 메모리 포인터
            SIZE,  # 실제 데이터 크기 (4) - 범위 검사에 사용됨
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (1개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (8개 스레드)
        )

        # 기대값을 저장할 호스트 메모리 버퍼를 생성하고 초기화합니다
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        # 이는 GPU의 비동기 실행 때문에 필요합니다
        ctx.synchronize()

        # 기대값 배열을 수동으로 계산하여 채웁니다
        # 각 요소는 원래 값(i)에 10을 더한 값이어야 합니다
        for i in range(SIZE):
            expected[i] = i + 10  # [10, 11, 12, 13]

        # GPU에서 계산된 결과를 호스트 메모리로 매핑하여 확인합니다
        with out.map_to_host() as out_host:
            print("out:", out_host)  # GPU에서 계산된 결과: [10, 11, 12, 13]
            print("expected:", expected)  # CPU에서 계산한 기대값: [10, 11, 12, 13]

            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
