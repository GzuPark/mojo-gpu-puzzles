from memory import UnsafePointer
from gpu import thread_idx, block_dim, block_idx  # GPU 스레드/블록 인덱스 및 차원 정보
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from testing import assert_equal

# ANCHOR: add
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias SIZE = 4  # 벡터의 크기 (4개의 요소)
alias BLOCKS_PER_GRID = 1  # GPU 그리드당 블록 수 (1개의 블록만 사용)
alias THREADS_PER_BLOCK = SIZE  # 블록당 스레드 수 (벡터 크기와 동일하게 4개)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)


# GPU 커널 함수: 두 벡터의 요소별 덧셈을 수행하는 함수
# 이 함수는 GPU의 각 스레드에서 병렬로 실행됩니다
fn add(
    output: UnsafePointer[Scalar[dtype]],  # 결과를 저장할 출력 배열의 메모리 포인터
    a: UnsafePointer[Scalar[dtype]],  # 첫 번째 입력 배열의 메모리 포인터
    b: UnsafePointer[Scalar[dtype]],  # 두 번째 입력 배열의 메모리 포인터
):
    # thread_idx.x: 현재 스레드의 x축 인덱스 (0, 1, 2, 3 중 하나)
    # 각 스레드는 고유한 인덱스를 가지며, 이를 통해 처리할 데이터 위치를 결정합니다
    i = thread_idx.x

    # FILL ME IN (roughly 1 line)
    # 벡터의 요소별 덧셈을 수행합니다
    # a[i]: 첫 번째 배열의 i번째 요소
    # b[i]: 두 번째 배열의 i번째 요소
    # a[i] + b[i]: 두 요소를 더한 결과
    # output[i]: 결과를 출력 배열의 i번째 위치에 저장
    output[i] = a[i] + b[i]


# ANCHOR_END: add


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # 이 컨텍스트는 GPU 메모리 할당, 데이터 복사, 커널 실행 등을 담당합니다
    with DeviceContext() as ctx:
        # GPU 메모리 버퍼들을 생성하고 0으로 초기화합니다 (메서드 체이닝 사용)
        # enqueue_create_buffer: GPU 메모리에 버퍼 생성
        # enqueue_fill(0): 버퍼를 0으로 초기화
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 출력 결과 버퍼
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 첫 번째 입력 버퍼
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 두 번째 입력 버퍼

        # 기대값을 저장할 호스트(CPU) 메모리 버퍼를 생성하고 초기화합니다
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)

        # 두 입력 배열을 동시에 호스트 메모리로 매핑하여 데이터를 초기화합니다
        # 복수의 컨텍스트 매니저를 콤마로 연결하여 동시에 사용할 수 있습니다
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            # 입력 데이터를 설정하고 동시에 기대값도 계산합니다
            for i in range(SIZE):
                a_host[i] = i  # 첫 번째 배열: [0, 1, 2, 3]
                b_host[i] = i  # 두 번째 배열: [0, 1, 2, 3]
                # 기대값: 각 위치에서 a[i] + b[i] = i + i = 2*i
                expected[i] = a_host[i] + b_host[i]  # [0, 2, 4, 6]

        # GPU 커널 함수를 실행합니다
        # add 함수를 GPU에서 병렬로 실행하도록 큐에 등록합니다
        # 세 개의 포인터를 전달: 출력, 첫 번째 입력, 두 번째 입력
        ctx.enqueue_function[add](
            out.unsafe_ptr(),  # 출력 버퍼의 메모리 포인터
            a.unsafe_ptr(),  # 첫 번째 입력 버퍼의 메모리 포인터
            b.unsafe_ptr(),  # 두 번째 입력 버퍼의 메모리 포인터
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (블록의 개수)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (블록당 스레드 개수)
        )

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        # 이 호출 없이는 GPU 작업이 완료되기 전에 결과를 확인할 수 있습니다
        ctx.synchronize()

        # GPU에서 계산된 결과를 호스트 메모리로 매핑하여 확인합니다
        with out.map_to_host() as out_host:
            print("out:", out_host)  # GPU에서 계산된 결과: [0, 2, 4, 6]
            print("expected:", expected)  # CPU에서 계산한 기대값: [0, 2, 4, 6]

            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
