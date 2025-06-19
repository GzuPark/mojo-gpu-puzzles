from memory import UnsafePointer
from gpu import thread_idx  # GPU 스레드의 인덱스 정보를 가져오기 위한 라이브러리
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from testing import assert_equal

# ANCHOR: add_10
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias SIZE = 4  # 벡터의 크기 (4개의 요소)
alias BLOCKS_PER_GRID = 1  # GPU 그리드당 블록 수 (1개의 블록만 사용)
alias THREADS_PER_BLOCK = SIZE  # 블록당 스레드 수 (벡터 크기와 동일하게 4개)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)


# GPU 커널 함수: 각 요소에 10을 더하는 함수
# 이 함수는 GPU의 각 스레드에서 실행됩니다
fn add_10(
    output: UnsafePointer[Scalar[dtype]],  # 출력 결과를 저장할 메모리 포인터
    a: UnsafePointer[Scalar[dtype]],  # 입력 데이터가 저장된 메모리 포인터
):
    # thread_idx.x: 현재 스레드의 x축 인덱스 (0, 1, 2, 3 중 하나)
    # 각 스레드는 자신의 고유한 인덱스를 가지며, 이를 통해 처리할 데이터를 구분합니다
    i = thread_idx.x

    # FILL ME IN (roughly 1 line)
    # 각 스레드가 담당하는 인덱스 i의 데이터를 처리합니다
    # a[i]: 입력 배열의 i번째 요소를 읽어옵니다
    # + 10.0: 읽어온 값에 10.0을 더합니다
    # output[i]: 결과를 출력 배열의 i번째 위치에 저장합니다
    output[i] = a[i] + 10.0


# ANCHOR_END: add_10


# 메인 함수: 프로그램의 진입점
def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # 이 컨텍스트는 GPU 메모리 할당, 데이터 복사, 커널 실행 등을 담당합니다
    with DeviceContext() as ctx:
        # GPU 메모리에 출력용 버퍼를 생성합니다
        # SIZE 개의 float32 데이터를 저장할 수 있는 공간을 할당합니다
        out = ctx.enqueue_create_buffer[dtype](SIZE)

        # 출력 버퍼를 0으로 초기화합니다
        # enqueue_fill(0): 비동기적으로 버퍼를 0으로 채우는 작업을 큐에 등록합니다
        out = out.enqueue_fill(0)

        # GPU 메모리에 입력용 버퍼를 생성합니다
        a = ctx.enqueue_create_buffer[dtype](SIZE)

        # 입력 버퍼도 0으로 초기화합니다
        a = a.enqueue_fill(0)

        # 입력 데이터를 초기화하기 위해 GPU 메모리를 호스트(CPU) 메모리로 매핑합니다
        # map_to_host(): GPU 메모리를 CPU에서 접근 가능하도록 매핑합니다
        with a.map_to_host() as a_host:
            # 입력 배열에 0, 1, 2, 3 값을 저장합니다
            for i in range(SIZE):
                a_host[i] = i  # i번째 위치에 i 값을 저장

        # GPU 커널 함수를 실행합니다
        # enqueue_function: 커널 함수를 GPU에서 실행하도록 큐에 등록합니다
        # [add_10]: 실행할 커널 함수를 지정합니다 (컴파일 타임 매개변수)
        # out.unsafe_ptr(): 출력 버퍼의 메모리 포인터를 전달합니다
        # a.unsafe_ptr(): 입력 버퍼의 메모리 포인터를 전달합니다
        # grid_dim=BLOCKS_PER_GRID: 그리드 차원 (블록의 개수) 설정
        # block_dim=THREADS_PER_BLOCK: 블록 차원 (블록당 스레드 개수) 설정
        ctx.enqueue_function[add_10](
            out.unsafe_ptr(),
            a.unsafe_ptr(),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # 기대값을 저장할 호스트 메모리 버퍼를 생성합니다
        # 이는 CPU 메모리에 생성되며, 결과 검증을 위해 사용됩니다
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)

        # 기대값 버퍼를 0으로 초기화합니다
        expected = expected.enqueue_fill(0)

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        # synchronize(): 큐에 등록된 모든 작업의 완료를 기다립니다
        ctx.synchronize()

        # 기대값 배열을 수동으로 계산하여 채웁니다
        # 각 요소는 원래 값(i)에 10을 더한 값이어야 합니다
        for i in range(SIZE):
            expected[i] = i + 10  # 0+10, 1+10, 2+10, 3+10

        # GPU에서 계산된 결과를 호스트 메모리로 매핑하여 확인합니다
        with out.map_to_host() as out_host:
            # 실제 결과와 기대값을 출력합니다
            print("out:", out_host)  # GPU에서 계산된 결과
            print("expected:", expected)  # 수동으로 계산한 기대값

            # 각 요소가 올바르게 계산되었는지 검증합니다
            for i in range(SIZE):
                # assert_equal: 두 값이 같은지 확인하고, 다르면 에러를 발생시킵니다
                assert_equal(out_host[i], expected[i])
