from gpu import thread_idx, block_dim, block_idx  # GPU 스레드 정보와 블록 차원 정보
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from layout import Layout, LayoutTensor  # 텐서의 메모리 레이아웃과 LayoutTensor 클래스
from testing import assert_equal

# ANCHOR: add_10_2d_layout_tensor
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias SIZE = 2  # 2D 배열의 크기 (2x2 배열 처리)
alias BLOCKS_PER_GRID = 1  # 그리드당 블록 개수 (1개 블록만 사용)
alias THREADS_PER_BLOCK = (3, 3)  # 블록당 스레드 개수 (3x3 = 9개 스레드, 데이터보다 많음)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)
# Row-major 레이아웃: 행 우선 순서로 메모리에 저장되는 2D 레이아웃
# [a, b]     메모리: [a, b, c, d]
# [c, d] →
alias layout = Layout.row_major(SIZE, SIZE)  # 2x2 크기의 row-major 레이아웃


# GPU 커널 함수: LayoutTensor를 사용한 2D 배열 처리
# LayoutTensor의 핵심 장점은 자연스러운 2D 인덱싱을 제공하는 것입니다
fn add_10_2d(
    output: LayoutTensor[mut=True, dtype, layout],  # 결과를 저장할 변경 가능한 2D 텐서
    a: LayoutTensor[mut=True, dtype, layout],  # 입력 데이터를 담은 2D 텐서
    size: Int,  # 배열의 실제 크기 (bounds checking용)
):
    # 2D 스레드 블록에서 현재 스레드의 위치를 가져옵니다
    # thread_idx.y: 현재 스레드의 행(row) 인덱스 (0, 1, 2)
    # thread_idx.x: 현재 스레드의 열(column) 인덱스 (0, 1, 2)
    row = thread_idx.y
    col = thread_idx.x

    # 스레드 안전성을 위한 경계 검사
    # 3x3 스레드 블록을 사용하지만 데이터는 2x2이므로 일부 스레드는 유효하지 않음
    # 유효한 스레드: (0,0), (0,1), (1,0), (1,1) - 총 4개
    # 무효한 스레드: (0,2), (1,2), (2,0), (2,1), (2,2) - 총 5개
    if row < size and col < size:
        # LayoutTensor의 핵심 장점: 자연스러운 2D 인덱싱
        # 기존 방식: output[row * size + col] = a[row * size + col] + 10.0
        # LayoutTensor: output[row, col] = a[row, col] + 10.0
        output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_2d_layout_tensor


def main():
    # GPU 디바이스 컨텍스트를 생성합니다 (context manager 사용)
    with DeviceContext() as ctx:
        # 출력용 GPU 메모리 버퍼를 생성하고 0으로 초기화
        # SIZE * SIZE = 2 * 2 = 4개의 float32 요소를 위한 1D 버퍼
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)

        # 1D 버퍼를 2D LayoutTensor로 래핑합니다
        # mut=True: 텐서 내용을 수정할 수 있도록 설정
        # layout: row-major 2x2 레이아웃으로 해석
        out_tensor = LayoutTensor[mut=True, dtype, layout](out_buf.unsafe_ptr())

        # LayoutTensor는 shape 정보를 제공합니다
        # shape[0](): 행의 개수 (2)
        # shape[1](): 열의 개수 (2)
        print("out shape:", out_tensor.shape[0](), "x", out_tensor.shape[1]())

        # 예상 결과를 저장할 호스트 메모리 버퍼 생성
        expected = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)

        # 입력 데이터용 GPU 메모리 버퍼를 생성하고 0으로 초기화
        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)

        # GPU 버퍼를 호스트 메모리에 매핑하여 데이터를 초기화합니다
        with a.map_to_host() as a_host:
            # 2D 배열을 1D로 저장: [0, 1, 2, 3]
            # 2x2 배열로 해석하면:
            # [0, 1]
            # [2, 3]
            for i in range(SIZE * SIZE):
                a_host[i] = i  # 입력 데이터: 0, 1, 2, 3
                expected[i] = a_host[i] + 10  # 예상 결과: 10, 11, 12, 13

        # 입력 버퍼를 LayoutTensor로 래핑합니다
        # 이제 a_tensor[0,0]=0, a_tensor[0,1]=1, a_tensor[1,0]=2, a_tensor[1,1]=3
        a_tensor = LayoutTensor[mut=True, dtype, layout](a.unsafe_ptr())

        # GPU 커널 함수를 실행합니다
        # add_10_2d: 각 요소에 10을 더하는 LayoutTensor 기반 커널
        # out_tensor: 결과를 저장할 2D 텐서
        # a_tensor: 입력 데이터를 담은 2D 텐서
        # SIZE: 배열 크기 (bounds checking용)
        # grid_dim: 1개 블록 사용
        # block_dim: (3,3) = 9개 스레드 (데이터 4개보다 많음, bounds checking 필요)
        ctx.enqueue_function[add_10_2d](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        ctx.synchronize()

        # GPU 메모리 결과를 호스트 메모리로 복사하여 확인합니다
        with out_buf.map_to_host() as out_buf_host:
            # out: [10.0, 11.0, 12.0, 13.0] (입력 [0,1,2,3]에 각각 10을 더한 결과)
            print("out:", out_buf_host)
            print("expected:", expected)

            for i in range(SIZE * SIZE):
                assert_equal(out_buf_host[i], expected[i])
