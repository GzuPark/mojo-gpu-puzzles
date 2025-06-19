from memory import UnsafePointer
from gpu import thread_idx, block_dim, block_idx  # GPU 스레드/블록 인덱스 및 차원 정보
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from testing import assert_equal

# ANCHOR: add_10_2d
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias SIZE = 2  # 2D 배열의 한 변의 크기 (2x2 매트릭스)
alias BLOCKS_PER_GRID = 1  # GPU 그리드당 블록 수 (1개의 블록만 사용)
alias THREADS_PER_BLOCK = (3, 3)  # 블록당 스레드 수 (3x3 = 9개) - 2D 배열보다 큼!
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)


# GPU 커널 함수: 2D 배열의 각 요소에 10을 더하는 함수
# 이 함수는 2차원 스레드 블록을 사용하여 2D 데이터를 병렬로 처리합니다
fn add_10_2d(
    output: UnsafePointer[Scalar[dtype]],  # 출력 결과를 저장할 메모리 포인터
    a: UnsafePointer[Scalar[dtype]],  # 입력 2D 배열이 저장된 메모리 포인터
    size: Int,  # 2D 배열의 한 변의 크기 (NxN 배열에서 N)
):
    # 2D 스레드 인덱스를 가져옵니다
    # thread_idx.y: 현재 스레드의 y축(행) 인덱스 (0~2 중 하나)
    # thread_idx.x: 현재 스레드의 x축(열) 인덱스 (0~2 중 하나)
    row = thread_idx.y  # 행 인덱스 (세로 방향)
    col = thread_idx.x  # 열 인덱스 (가로 방향)

    # FILL ME IN (roughly 2 lines)
    # 2D 범위 검사: 행과 열 모두 유효한 범위 내에 있는지 확인
    # 3x3 스레드 그리드에서 2x2 데이터를 처리하므로 일부 스레드는 범위를 벗어남
    # 유효한 스레드: (0,0), (0,1), (1,0), (1,1)
    # 무효한 스레드: (0,2), (1,2), (2,0), (2,1), (2,2)
    if row < size and col < size:
        # 2D 인덱스를 1D 메모리 인덱스로 변환 (row-major 방식)
        # row * size + col: 2D 좌표 (row, col)을 1D 인덱스로 변환
        # 예: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3
        # a[...]: 입력 배열의 해당 위치 값을 읽어옵니다
        # + 10.0: 읽어온 값에 10.0을 더합니다
        # output[...]: 결과를 출력 배열의 해당 위치에 저장합니다
        output[row * size + col] = a[row * size + col] + 10.0


# ANCHOR_END: add_10_2d


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # 이 컨텍스트는 GPU 메모리 할당, 데이터 복사, 커널 실행 등을 담당합니다
    with DeviceContext() as ctx:
        # 2D 배열을 위한 GPU 메모리 버퍼들을 생성합니다
        # SIZE * SIZE: 2x2 = 4개의 요소를 가진 1차원 배열로 저장
        out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(
            0
        )  # 출력 결과 버퍼

        # 기대값을 저장할 호스트 메모리 버퍼를 생성합니다
        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE  # 2x2 = 4개 요소
        ).enqueue_fill(0)

        # 입력 2D 배열을 위한 GPU 메모리 버퍼를 생성합니다
        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)

        # 입력 2D 배열을 초기화하기 위해 GPU 메모리를 호스트 메모리로 매핑합니다
        with a.map_to_host() as a_host:
            # row-major 방식으로 2D 배열을 1D 메모리에 저장합니다
            # 2D 배열:        1D 메모리:
            # [0, 1]    →    [0, 1, 2, 3]
            # [2, 3]
            for i in range(SIZE):  # 행 (row) 순회: i = 0, 1
                for j in range(SIZE):  # 열 (column) 순회: j = 0, 1
                    # 2D 인덱스 (i, j)를 1D 인덱스로 변환: i * SIZE + j
                    # (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3
                    a_host[i * SIZE + j] = i * SIZE + j  # 값: 0, 1, 2, 3
                    # 기대값을 동시에 계산: 각 요소에 10을 더한 값
                    expected[i * SIZE + j] = (
                        a_host[i * SIZE + j] + 10
                    )  # 값: 10, 11, 12, 13

        # GPU 커널 함수를 실행합니다
        # 2D 스레드 블록 (3x3)을 사용하여 2D 데이터 (2x2)를 처리합니다
        ctx.enqueue_function[add_10_2d](
            out.unsafe_ptr(),  # 출력 버퍼의 메모리 포인터
            a.unsafe_ptr(),  # 입력 2D 배열의 메모리 포인터
            SIZE,  # 2D 배열의 한 변의 크기 (2)
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (1개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (3x3 = 9개 스레드)
        )

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        ctx.synchronize()

        # GPU에서 계산된 결과를 호스트 메모리로 매핑하여 확인합니다
        with out.map_to_host() as out_host:
            print("out:", out_host)  # GPU 결과: [10.0, 11.0, 12.0, 13.0]
            print("expected:", expected)  # 기대값: [10.0, 11.0, 12.0, 13.0]

            # 2D 배열의 각 요소가 올바르게 계산되었는지 검증합니다
            for i in range(SIZE):  # 행별로 검증
                for j in range(SIZE):  # 열별로 검증
                    # 2D 인덱스를 1D 인덱스로 변환하여 비교
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j])
