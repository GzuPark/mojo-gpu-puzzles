from gpu import thread_idx, block_dim, block_idx  # GPU 스레드 정보와 블록 차원 정보
from gpu.host import DeviceContext, HostBuffer  # GPU 디바이스 컨텍스트와 호스트 버퍼
from layout import Layout, LayoutTensor  # 텐서의 메모리 레이아웃과 LayoutTensor 클래스
from testing import assert_equal

# ANCHOR: broadcast_add_layout_tensor
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias SIZE = 2  # 2D 배열의 크기 (2x2 배열 처리)
alias BLOCKS_PER_GRID = 1  # 그리드당 블록 개수 (1개 블록만 사용)
alias THREADS_PER_BLOCK = (3, 3)  # 블록당 스레드 개수 (3x3 = 9개 스레드, 데이터보다 많음)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)

# 각 텐서의 레이아웃을 정의합니다 (브로드캐스트를 위한 서로 다른 형태)
alias out_layout = Layout.row_major(SIZE, SIZE)  # 출력: 2x2 행렬
alias a_layout = Layout.row_major(1, SIZE)  # 벡터 a: 1x2 행렬 (행 벡터, 열 방향 브로드캐스트)
alias b_layout = Layout.row_major(SIZE, 1)  # 벡터 b: 2x1 행렬 (열 벡터, 행 방향 브로드캐스트)


# GPU 커널 함수: LayoutTensor를 사용한 브로드캐스트 덧셈 연산
# 서로 다른 형태의 LayoutTensor들을 브로드캐스트하여 더합니다
# 예: a(1x2) + b(2x1) → output(2x2)
#     [[0, 2]] + [[0],   → [[0+0, 2+0],   → [[0, 2],
#                 [3]]      [0+3, 2+3]]      [3, 5]]
fn broadcast_add[
    out_layout: Layout,  # 출력 텐서의 레이아웃 (컴파일 타임 매개변수)
    a_layout: Layout,  # 벡터 a의 레이아웃 (컴파일 타임 매개변수)
    b_layout: Layout,  # 벡터 b의 레이아웃 (컴파일 타임 매개변수)
](
    output: LayoutTensor[mut=True, dtype, out_layout],  # 결과를 저장할 2x2 텐서
    a: LayoutTensor[mut=False, dtype, a_layout],  # 1x2 행 벡터 (열 방향 브로드캐스트)
    b: LayoutTensor[mut=False, dtype, b_layout],  # 2x1 열 벡터 (행 방향 브로드캐스트)
    size: Int,  # 배열의 크기 (bounds checking용)
):
    # 2D 스레드 블록에서 현재 스레드의 위치를 가져옵니다
    # ⚠️ 주의: thread_idx의 x, y와 row, col의 대응 관계를 정확히 이해해야 합니다!
    # thread_idx.y → row (행 인덱스): 세로 방향 위치 (0, 1, 2)
    # thread_idx.x → col (열 인덱스): 가로 방향 위치 (0, 1, 2)
    row = thread_idx.y  # 행 인덱스 (Y축 = 세로 방향)
    col = thread_idx.x  # 열 인덱스 (X축 = 가로 방향)

    # 스레드 안전성을 위한 경계 검사
    # 3x3 스레드 블록을 사용하지만 데이터는 2x2이므로 일부 스레드는 유효하지 않음
    if row < size and col < size:
        # LayoutTensor 브로드캐스트 덧셈의 핵심 로직:
        # ⚠️ 인덱스 순서 주의: a[0, col] + b[row, 0]
        # - a[0, col]: 1x2 행 벡터에서 (0행, col열) 요소 → 열 방향으로 반복
        # - b[row, 0]: 2x1 열 벡터에서 (row행, 0열) 요소 → 행 방향으로 반복
        # - output[row, col]: 자연스러운 2D 인덱싱으로 결과 저장
        output[row, col] = a[0, col] + b[row, 0]


# ANCHOR_END: broadcast_add_layout_tensor
def main():
    # GPU 디바이스 컨텍스트를 생성합니다 (context manager 사용)
    with DeviceContext() as ctx:
        # 출력용 GPU 메모리 버퍼를 생성하고 0으로 초기화 (2x2 = 4개 요소)
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(0)

        # 1D 버퍼를 2x2 LayoutTensor로 래핑합니다
        out_tensor = LayoutTensor[mut=True, dtype, out_layout](
            out_buf.unsafe_ptr()
        )

        # LayoutTensor는 shape 정보를 제공합니다
        print("out shape:", out_tensor.shape[0](), "x", out_tensor.shape[1]())

        # 예상 결과를 저장할 호스트 메모리 버퍼와 텐서 생성
        expected_buf = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)
        expected_tensor = LayoutTensor[mut=True, dtype, out_layout](
            expected_buf.unsafe_ptr()
        )

        # 입력 벡터들을 위한 GPU 메모리 버퍼 생성
        # a: 1x2 행 벡터로 사용될 1D 버퍼 (크기: SIZE)
        # b: 2x1 열 벡터로 사용될 1D 버퍼 (크기: SIZE)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)

        # GPU 버퍼들을 호스트 메모리에 매핑하여 데이터를 초기화합니다
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            # 입력 벡터들을 초기화합니다
            for i in range(SIZE):
                a_host[i] = i * 2  # a = [0, 2] (1x2 행 벡터로 브로드캐스트)
                b_host[i] = i * 3  # b = [0, 3] (2x1 열 벡터로 브로드캐스트)

            # 예상 결과를 계산합니다 (LayoutTensor 브로드캐스트 덧셈)
            # ⚠️ 인덱스 순서 주의: i는 행, j는 열
            # expected_tensor[i, j] = a_host[j] + b_host[i]
            # 브로드캐스트 결과:
            # [[a[0]+b[0], a[1]+b[0]]   [[0+0, 2+0]]   [[0, 2]]
            #  [a[0]+b[1], a[1]+b[1]]] = [[0+3, 2+3]] = [[3, 5]]
            for i in range(SIZE):  # i: 행 인덱스
                for j in range(SIZE):  # j: 열 인덱스
                    expected_tensor[i, j] = a_host[j] + b_host[i]

        # 1D 버퍼들을 각각 다른 형태의 LayoutTensor로 래핑합니다
        # a_tensor: 1x2 행 벡터 (열 방향으로 브로드캐스트됨)
        # b_tensor: 2x1 열 벡터 (행 방향으로 브로드캐스트됨)
        a_tensor = LayoutTensor[dtype, a_layout](a.unsafe_ptr())
        b_tensor = LayoutTensor[dtype, b_layout](b.unsafe_ptr())

        # GPU 커널 함수를 실행합니다
        # broadcast_add: 서로 다른 형태의 LayoutTensor들을 브로드캐스트하여 덧셈 수행
        # out_tensor: 결과를 저장할 2x2 텐서
        # a_tensor: 1x2 행 벡터 (열 방향 브로드캐스트)
        # b_tensor: 2x1 열 벡터 (행 방향 브로드캐스트)
        # SIZE: 배열 크기 (bounds checking용)
        # 컴파일 타임 매개변수: [out_layout, a_layout, b_layout]
        ctx.enqueue_function[broadcast_add[out_layout, a_layout, b_layout]](
            out_tensor,
            a_tensor,
            b_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        ctx.synchronize()

        # GPU 메모리 결과를 호스트 메모리로 복사하여 확인합니다
        with out_buf.map_to_host() as out_buf_host:
            # 결과를 출력합니다
            # out: [0.0, 2.0, 3.0, 5.0] (1D 배열로 저장된 2x2 브로드캐스트 결과)
            # 2D로 해석하면: [[0, 2], [3, 5]]
            print("out:", out_buf_host)
            print("expected:", expected_buf)

            for i in range(SIZE):  # i: 행 인덱스
                for j in range(SIZE):  # j: 열 인덱스
                    assert_equal(
                        out_buf_host[i * SIZE + j], expected_buf[i * SIZE + j]
                    )
