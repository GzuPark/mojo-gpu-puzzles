from gpu import thread_idx, block_idx, block_dim  # GPU 스레드/블록 인덱스 및 차원 정보
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from layout import Layout, LayoutTensor
from testing import assert_equal

# ANCHOR: add_10_blocks_2d_layout_tensor
# 프로그램 설정값들 (컴파일 타임 상수)
alias SIZE = 5  # 정사각 행렬의 한 변 크기 (5x5 = 25개 요소)
alias BLOCKS_PER_GRID = (2, 2)  # 그리드 구성: 2x2 = 4개 블록
alias THREADS_PER_BLOCK = (3, 3)  # 각 블록 구성: 3x3 = 9개 스레드
alias dtype = DType.float32  # 부동소수점 32비트 데이터 타입

# LayoutTensor 설정: 행 우선 순서(Row-Major) 레이아웃 정의
alias out_layout = Layout.row_major(SIZE, SIZE)  # 출력 텐서 레이아웃 (5x5)
alias a_layout = Layout.row_major(SIZE, SIZE)  # 입력 텐서 레이아웃 (5x5)


# 이 커널은 LayoutTensor의 고급 기능을 활용하여 2D 행렬을 자연스럽게 처리합니다
# 핵심 장점: 수동 인덱스 계산(row*size+col) 없이 직관적인 [row,col] 접근 가능
fn add_10_blocks_2d[
    out_layout: Layout,  # 출력 텐서의 메모리 레이아웃 (컴파일 타임 매개변수)
    a_layout: Layout,  # 입력 텐서의 메모리 레이아웃 (컴파일 타임 매개변수)
](
    output: LayoutTensor[mut=True, dtype, out_layout],  # 수정 가능한 출력 텐서
    a: LayoutTensor[mut=False, dtype, a_layout],  # 읽기 전용 입력 텐서
    size: Int,  # 정사각 행렬의 한 변 길이
):
    # 각 스레드가 담당할 행렬의 전역 위치(row, col)를 계산합니다
    #
    # row = block_dim.y × block_idx.y + thread_idx.y
    # col = block_dim.x × block_idx.x + thread_idx.x
    #
    # 🔸 블록 (0,0) - 좌상단 영역:
    #   시작 위치: (0×3, 0×3) = (0,0)
    #   스레드 (0,0) → 글로벌 (0,0)  스레드 (0,1) → 글로벌 (0,1)  스레드 (0,2) → 글로벌 (0,2)
    #   스레드 (1,0) → 글로벌 (1,0)  스레드 (1,1) → 글로벌 (1,1)  스레드 (1,2) → 글로벌 (1,2)
    #   스레드 (2,0) → 글로벌 (2,0)  스레드 (2,1) → 글로벌 (2,1)  스레드 (2,2) → 글로벌 (2,2)
    #
    # 🔸 블록 (1,0) - 우상단 영역:
    #   시작 위치: (0×3, 1×3) = (0,3)
    #   스레드 (0,0) → 글로벌 (0,3)  스레드 (0,1) → 글로벌 (0,4)  스레드 (0,2) → 글로벌 (0,5) ❌범위초과
    #   스레드 (1,0) → 글로벌 (1,3)  스레드 (1,1) → 글로벌 (1,4)  스레드 (1,2) → 글로벌 (1,5) ❌범위초과
    #   스레드 (2,0) → 글로벌 (2,3)  스레드 (2,1) → 글로벌 (2,4)  스레드 (2,2) → 글로벌 (2,5) ❌범위초과
    #
    # 🔸 블록 (0,1) - 좌하단 영역:
    #   시작 위치: (1×3, 0×3) = (3,0)
    #   스레드 (0,0) → 글로벌 (3,0)  스레드 (0,1) → 글로벌 (3,1)  스레드 (0,2) → 글로벌 (3,2)
    #   스레드 (1,0) → 글로벌 (4,0)  스레드 (1,1) → 글로벌 (4,1)  스레드 (1,2) → 글로벌 (4,2)
    #   스레드 (2,0) → 글로벌 (5,0) ❌범위초과  스레드 (2,1) → 글로벌 (5,1) ❌범위초과  스레드 (2,2) → 글로벌 (5,2) ❌범위초과
    #
    # 🔸 블록 (1,1) - 우하단 영역:
    #   시작 위치: (1×3, 1×3) = (3,3)
    #   스레드 (0,0) → 글로벌 (3,3)  스레드 (0,1) → 글로벌 (3,4)  스레드 (0,2) → 글로벌 (3,5) ❌범위초과
    #   스레드 (1,0) → 글로벌 (4,3)  스레드 (1,1) → 글로벌 (4,4)  스레드 (1,2) → 글로벌 (4,5) ❌범위초과
    #   스레드 (2,0) → 글로벌 (5,3) ❌범위초과  스레드 (2,1) → 글로벌 (5,4) ❌범위초과  스레드 (2,2) → 글로벌 (5,5) ❌범위초과
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x

    # 계산된 좌표가 실제 행렬 범위 내에 있는지 확인합니다
    # • 총 스레드 수: 4블록 × 9스레드 = 36개
    # • 실제 데이터: 5×5 = 25개 요소
    # • 차이: 36 - 25 = 11개 스레드가 "빈 공간"을 가리킴
    if row < size and col < size:
        # 1. [row, col] 접근 시 자동으로 1D 인덱스 계산
        # 2. 레이아웃 정보를 사용하여 올바른 메모리 위치 결정
        # 3. 컴파일 타임에 최적화되어 성능 손실 없음
        output[row, col] = a[row, col] + 10.0


# ANCHOR_END: add_10_blocks_2d_layout_tensor


def main():
    # GPU 디바이스 컨텍스트를 생성하여 모든 GPU 작업을 관리합니다
    # 이 컨텍스트는 메모리 할당, 데이터 전송, 커널 실행을 담당합니다
    with DeviceContext() as ctx:
        # 5×5 행렬을 위한 GPU 메모리 버퍼를 생성하고 LayoutTensor로 래핑합니다
        # 출력용 GPU 버퍼 및 LayoutTensor 생성
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(
            0
        )  # GPU 버퍼 (초기값: 0.0)
        out_tensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin](
            out_buf.unsafe_ptr()  # 1D 버퍼를 2D LayoutTensor로 래핑
        )

        # 기대값 저장용 호스트 버퍼 생성
        expected_buf = ctx.enqueue_create_host_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(
            1
        )  # CPU 버퍼 (초기값: 1.0)

        # 입력용 GPU 버퍼 및 LayoutTensor 생성
        a = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(
            1
        )  # GPU 버퍼 (초기값: 1.0)
        a_tensor = LayoutTensor[dtype, a_layout, MutableAnyOrigin](
            a.unsafe_ptr()  # 1D 버퍼를 2D LayoutTensor로 래핑
        )

        # 다중 2D 블록을 사용하여 5×5 행렬을 병렬 처리합니다
        #
        # 📊 리소스 분석 (p07.mojo와 동일):
        # • 처리할 데이터: 5×5 = 25개 요소
        # • 사용할 블록: 2×2 = 4개 블록
        # • 블록당 스레드: 3×3 = 9개 스레드
        # • 총 스레드: 4×9 = 36개 스레드
        # • 오버헤드: 36-25 = 11개 스레드 (30% 오버헤드)
        #
        # 🎯 LayoutTensor의 장점:
        # • 커널 함수에서 자연스러운 [row, col] 인덱싱 사용
        # • 컴파일 타임 레이아웃 최적화
        # • 타입 안전성 보장
        # • 수동 인덱스 계산 오류 방지
        ctx.enqueue_function[add_10_blocks_2d[out_layout, a_layout]](
            out_tensor,  # 출력 LayoutTensor
            a_tensor,  # 입력 LayoutTensor
            SIZE,  # 행렬 크기 (5)
            grid_dim=BLOCKS_PER_GRID,  # 그리드: 2×2 블록
            block_dim=THREADS_PER_BLOCK,  # 블록: 3×3 스레드
        )

        # 모든 GPU 커널 실행이 완료될 때까지 CPU가 대기
        ctx.synchronize()

        # GPU 결과와 비교하기 위해 CPU에서 동일한 연산을 수행합니다
        # 여기서도 LayoutTensor의 자연스러운 2D 인덱싱을 활용합니다
        expected_tensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin](
            expected_buf.unsafe_ptr()  # 호스트 버퍼를 LayoutTensor로 래핑
        )

        # LayoutTensor를 사용한 기대값 계산
        # 🆚 비교: 기존 방식 vs LayoutTensor 방식
        # 기존: expected[i * SIZE + j] += 10  ← 수동 인덱스 계산
        # 현재: expected_tensor[i, j] += 10   ← 직관적 2D 인덱싱
        for i in range(SIZE):  # 행 인덱스 (0~4)
            for j in range(SIZE):  # 열 인덱스 (0~4)
                expected_tensor[i, j] += 10

        # GPU 메모리의 결과를 CPU로 복사하여 검증합니다
        with out_buf.map_to_host() as out_buf_host:
            # LayoutTensor를 사용한 결과 출력
            # GPU 결과도 LayoutTensor로 래핑하여 2D 형태로 표시
            print(
                "out:",
                LayoutTensor[dtype, out_layout, MutableAnyOrigin](
                    out_buf_host.unsafe_ptr()  # 호스트로 복사된 GPU 결과를 LayoutTensor로 래핑
                ),
            )
            print("expected:", expected_tensor)

            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(
                        out_buf_host[i * SIZE + j], expected_buf[i * SIZE + j]
                    )
