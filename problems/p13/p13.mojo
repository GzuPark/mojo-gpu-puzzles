from sys import sizeof
from testing import assert_equal
from gpu.host import DeviceContext

# ANCHOR: axis_sum
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb

# Axis Sum (축별 합계) - 2D 행렬 연산
#
# 다차원 배열(행렬)에서 특정 축(axis)을 따라 합계를 계산하는 연산입니다.
# 이 문제에서는 각 행(row)별로 합계를 계산합니다.
#
# 예시: 4×6 행렬
# 입력 행렬:
# Row 0: [0,  1,  2,  3,  4,  5 ] → 합계: 15
# Row 1: [6,  7,  8,  9,  10, 11] → 합계: 51
# Row 2: [12, 13, 14, 15, 16, 17] → 합계: 87
# Row 3: [18, 19, 20, 21, 22, 23] → 합계: 123
#
# 출력: [15, 51, 87, 123]
#
# 핵심 개념들:
# 1. 2D Layout: 다차원 텐서 레이아웃 처리
# 2. Block Coordinate Mapping: 블록 좌표를 행렬 차원에 매핑
# 3. Axis-wise Reduction: 특정 축을 따른 병렬 리덕션
# 4. Multi-dimensional Indexing: 다차원 인덱싱 패턴

alias TPB = 8  # Threads Per Block
alias BATCH = 4  # 행렬의 행 개수 (배치 크기)
alias SIZE = 6  # 행렬의 열 개수 (각 행의 크기)
alias BLOCKS_PER_GRID = (1, BATCH)  # 새로운 개념: 2D 그리드 구성 (1×4)
alias THREADS_PER_BLOCK = (TPB, 1)  # 1D 스레드 블록 (8×1)
alias dtype = DType.float32

# Multi-dimensional Layout (다차원 레이아웃)
#
# 2D 텐서를 위한 레이아웃 정의:
# - in_layout: 입력 행렬 (BATCH×SIZE = 4×6)
# - out_layout: 출력 벡터 (BATCH×1 = 4×1)
#
# Layout.row_major(BATCH, SIZE)의 의미:
# - 행 우선(row-major) 메모리 배치
# - 연속된 메모리에 행별로 저장
# - 메모리 레이아웃: [row0_data][row1_data][row2_data][row3_data]
alias in_layout = Layout.row_major(BATCH, SIZE)  # 4×6 입력 행렬
alias out_layout = Layout.row_major(BATCH, 1)  # 4×1 출력 벡터


# GPU 커널 함수: Axis Sum 연산
#
# Block Coordinate Mapping (블록 좌표 매핑)
#
# 그리드 구성: BLOCKS_PER_GRID = (1, BATCH) = (1, 4)
# - x 차원: 1개 블록 (모든 행이 동일한 x 좌표)
# - y 차원: 4개 블록 (각 행마다 하나의 블록)
#
# 블록-행 매핑:
# Block(0,0) → Row 0 처리
# Block(0,1) → Row 1 처리
# Block(0,2) → Row 2 처리
# Block(0,3) → Row 3 처리
#
# 이 방식의 장점:
# 1. 각 행이 독립적으로 병렬 처리됨
# 2. 블록 간 동기화 불필요
# 3. 메모리 접근 패턴 최적화
# 4. 확장성 (행 개수에 따라 블록 수 조정 가능)
fn axis_sum[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # 다차원 블록 인덱싱
    #
    # batch = block_idx.y: y 차원의 블록 인덱스를 사용하여 처리할 행 결정
    # - Block(0,0): batch = 0 → Row 0 처리
    # - Block(0,1): batch = 1 → Row 1 처리
    # - Block(0,2): batch = 2 → Row 2 처리
    # - Block(0,3): batch = 3 → Row 3 처리
    batch = block_idx.y

    # FILL ME IN (roughly 15 lines)

    # 공유 메모리 할당
    # 각 블록이 하나의 행을 처리하므로 TPB 크기의 공유 메모리 필요
    cache = tb[dtype]().row_major[TPB]().shared().alloc()

    # 2D 텐서 인덱싱과 데이터 로딩
    #
    # 시각화 - 각 블록의 데이터 로딩 패턴:
    # Block(0,0): [T0,T1,T2,T3,T4,T5,T6,T7] → Row 0: [0,1,2,3,4,5]
    # Block(0,1): [T0,T1,T2,T3,T4,T5,T6,T7] → Row 1: [6,7,8,9,10,11]
    # Block(0,2): [T0,T1,T2,T3,T4,T5,T6,T7] → Row 2: [12,13,14,15,16,17]
    # Block(0,3): [T0,T1,T2,T3,T4,T5,T6,T7] → Row 3: [18,19,20,21,22,23]
    #
    # 각 행은 해당하는 블록에 의해 처리됩니다 (grid_dim=(1, BATCH))
    #
    # LayoutTensor의 2D 인덱싱: a[batch, local_i]
    # - batch: 행 인덱스 (block_idx.y로 결정)
    # - local_i: 열 인덱스 (thread_idx.x로 결정)
    #
    # 메모리 접근 패턴:
    # - Block 0: a[0,0], a[0,1], a[0,2], a[0,3], a[0,4], a[0,5]
    # - Block 1: a[1,0], a[1,1], a[1,2], a[1,3], a[1,4], a[1,5]
    # - Block 2: a[2,0], a[2,1], a[2,2], a[2,3], a[2,4], a[2,5]
    # - Block 3: a[3,0], a[3,1], a[3,2], a[3,3], a[3,4], a[3,5]

    if local_i < size:
        # 유효한 데이터 로드: 2D 인덱싱 사용
        cache[local_i] = a[batch, local_i]
    else:
        # 패딩 처리: 나중에 리덕션을 위해 0으로 초기화
        # TPB(8) > SIZE(6)이므로 Thread 6, 7은 패딩 값 0을 저장
        cache[local_i] = 0

    barrier()  # 모든 스레드의 데이터 로딩 완료 대기

    # Parallel Reduction with Halving Stride (반분 보폭 병렬 리덕션)
    #
    # 이전 Prefix Sum과 다른 리덕션 패턴:
    # Prefix Sum: offset을 2배씩 증가 (1→2→4)
    # Axis Sum: stride를 반으로 감소 (4→2→1)
    #
    # 리덕션 과정 시각화 (Block 0 예시):
    # 초기:     [0, 1, 2, 3, 4, 5, 0, 0]
    # Stride 4: [4, 5, 6, 7, 4, 5, 0, 0]  (cache[i] += cache[i+4])
    # Stride 2: [10,12, 6, 7, 4, 5, 0, 0]  (cache[i] += cache[i+2])
    # Stride 1: [15,12, 6, 7, 4, 5, 0, 0]  (cache[i] += cache[i+1])
    # 최종 결과: cache[0] = 15 (Row 0의 합계)

    stride = TPB // 2  # 시작 보폭: 8 // 2 = 4

    while stride > 0:
        # Race Condition 방지를 위한 2단계 접근법
        # 읽기 단계: 필요한 값을 지역 변수에 저장
        var temp_val: output.element_type = 0
        if local_i < stride:
            temp_val = cache[local_i + stride]

        barrier()  # 모든 읽기 완료 대기

        # 쓰기 단계: 읽은 값을 사용하여 합계 계산
        if local_i < stride:
            cache[local_i] += temp_val

        barrier()  # 모든 쓰기 완료 대기
        stride //= 2  # 보폭을 절반으로 감소

    # 2D 출력 인덱싱
    #
    # 각 블록의 Thread 0이 해당 배치(행)의 합계를 저장
    # output[batch, 0]: 2D 출력 텐서의 (batch, 0) 위치에 저장
    #
    # 결과 저장:
    # - Block 0, Thread 0: output[0, 0] = 15
    # - Block 1, Thread 0: output[1, 0] = 51
    # - Block 2, Thread 0: output[2, 0] = 87
    # - Block 3, Thread 0: output[3, 0] = 123
    if local_i == 0:
        output[batch, 0] = cache[0]


# ANCHOR_END: axis_sum


def main():
    with DeviceContext() as ctx:
        # 메모리 버퍼 생성
        # 출력: BATCH개의 스칼라 값 (각 행의 합계)
        # 입력: BATCH×SIZE 크기의 2D 행렬
        out = ctx.enqueue_create_buffer[dtype](BATCH).enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](BATCH * SIZE).enqueue_fill(0)

        # 입력 데이터 초기화: 연속된 정수 값
        # Row 0: [0, 1, 2, 3, 4, 5]
        # Row 1: [6, 7, 8, 9, 10, 11]
        # Row 2: [12, 13, 14, 15, 16, 17]
        # Row 3: [18, 19, 20, 21, 22, 23]
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    inp_host[row * SIZE + col] = row * SIZE + col

        # LayoutTensor 생성: 다차원 레이아웃 적용
        #
        # 다차원 LayoutTensor 생성
        # - out_tensor: (BATCH, 1) 형태의 2D 출력 텐서
        # - inp_tensor: (BATCH, SIZE) 형태의 2D 입력 텐서
        #
        # 1D 버퍼를 2D LayoutTensor로 해석:
        # - 물리적 메모리: 연속된 1D 배열
        # - 논리적 구조: 2D 행렬로 해석
        # - 레이아웃이 메모리 접근 패턴을 결정
        out_tensor = LayoutTensor[mut=False, dtype, out_layout](
            out.unsafe_ptr()
        )
        inp_tensor = LayoutTensor[mut=False, dtype, in_layout](inp.unsafe_ptr())

        # GPU 커널 실행
        #
        # 2D 그리드 실행 구성
        # grid_dim=BLOCKS_PER_GRID=(1, BATCH): 1×4 블록 그리드
        # - x 차원: 1개 블록
        # - y 차원: 4개 블록 (각 행마다 하나)
        #
        # 실행 패턴:
        # - 4개 블록이 동시에 실행
        # - 각 블록이 하나의 행을 독립적으로 처리
        # - 블록 간 동기화 불필요 (완전 병렬)
        ctx.enqueue_function[axis_sum[in_layout, out_layout]](
            out_tensor,
            inp_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,  # (1, 4) - 2D 그리드
            block_dim=THREADS_PER_BLOCK,  # (8, 1) - 1D 블록
        )

        # CPU 참조 구현 (검증용)
        expected = ctx.enqueue_create_host_buffer[dtype](BATCH).enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    expected[row] += inp_host[row * SIZE + col]

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out)  # GPU 결과
            print("expected:", expected)  # CPU 참조 결과
            for i in range(BATCH):
                assert_equal(out_host[i], expected[i])
