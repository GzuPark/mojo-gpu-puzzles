from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from math import log2  # 로그 계산 함수 (병렬 알고리즘의 반복 횟수 계산용)
from testing import assert_equal

# ANCHOR: prefix_sum_simple
# Prefix Sum (누적 합, Scan 연산)
# 입력 배열의 각 위치에서 그 위치까지의 모든 원소들의 누적 합을 계산하는 연산입니다.
#
# 예시: 입력 [0, 1, 2, 3, 4, 5, 6, 7]
# - output[0] = 0                           = 0
# - output[1] = 0 + 1                       = 1
# - output[2] = 0 + 1 + 2                   = 3
# - output[3] = 0 + 1 + 2 + 3               = 6
# - output[4] = 0 + 1 + 2 + 3 + 4           = 10
# - output[5] = 0 + 1 + 2 + 3 + 4 + 5       = 15
# - output[6] = 0 + 1 + 2 + 3 + 4 + 5 + 6   = 21
# - output[7] = 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
#
# 수학적 표현: output[i] = Σ(k=0 to i) input[k]
#
# 순차 알고리즘 vs 병렬 알고리즘:
# - 순차: O(n) 시간, 각 원소를 차례로 더함
# - 병렬: O(log n) 시간, 트리 기반 병렬 리덕션 사용
#
# 병렬 Prefix Sum의 핵심 아이디어:
# 각 스레드가 동시에 작업하되, 단계별로 거리(offset)를 늘려가며
# 이전 단계의 결과를 현재 위치에 누적하는 방식입니다.

# Simple Case 구성 상수들
alias TPB = 8  # Threads Per Block
alias SIZE = 8  # 입력 배열 크기 (단일 블록으로 처리 가능한 크기)
alias BLOCKS_PER_GRID = (1, 1)  # 단일 블록만 사용
alias THREADS_PER_BLOCK = (TPB, 1)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)


# GPU 커널 함수: Simple Case Prefix Sum
#
# Simple Case의 특징:
# 1. 단일 블록 내에서 모든 데이터 처리
# 2. 공유 메모리만으로 충분 (블록 간 통신 불필요)
# 3. log₂(TPB) = 3번의 반복으로 완료
# 4. 기본적인 병렬 Prefix Sum 알고리즘 학습에 적합
fn prefix_sum_simple[
    layout: Layout
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 18 lines)

    # 모든 입력 데이터를 공유 메모리에 로드하여 빠른 접근 보장
    shared = tb[dtype]().row_major[TPB]().shared().alloc()

    # 1단계: 입력 데이터를 공유 메모리로 로드
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()  # 모든 데이터 로딩 완료 대기

    # 2단계: 병렬 Prefix Sum 알고리즘 수행
    #
    # 핵심 알고리즘: Up-Sweep (Reduce) 방식의 Prefix Sum
    # 각 반복에서 offset을 2배씩 증가시키며 누적 합을 계산합니다.
    #
    # 시각적 표현 (입력: [0,1,2,3,4,5,6,7]):
    #
    # 초기 상태:     [0, 1, 2, 3, 4, 5, 6, 7]
    #
    # Iteration 1 (offset=1):
    # Thread 1: shared[1] += shared[0]  → 1 + 0 = 1
    # Thread 2: shared[2] += shared[1]  → 2 + 1 = 3
    # Thread 3: shared[3] += shared[2]  → 3 + 2 = 5
    # Thread 4: shared[4] += shared[3]  → 4 + 3 = 7
    # Thread 5: shared[5] += shared[4]  → 5 + 4 = 9
    # Thread 6: shared[6] += shared[5]  → 6 + 5 = 11
    # Thread 7: shared[7] += shared[6]  → 7 + 6 = 13
    # 결과:          [0, 1, 3, 5, 7, 9, 11, 13]
    #
    # Iteration 2 (offset=2):
    # Thread 2: shared[2] += shared[0]  → 3 + 0 = 3
    # Thread 3: shared[3] += shared[1]  → 5 + 1 = 6
    # Thread 4: shared[4] += shared[2]  → 7 + 3 = 10
    # Thread 5: shared[5] += shared[3]  → 9 + 5 = 14
    # Thread 6: shared[6] += shared[4]  → 11 + 7 = 18
    # Thread 7: shared[7] += shared[5]  → 13 + 9 = 22
    # 결과:          [0, 1, 3, 6, 10, 14, 18, 22]
    #
    # Iteration 3 (offset=4):
    # Thread 4: shared[4] += shared[0]  → 10 + 0 = 10
    # Thread 5: shared[5] += shared[1]  → 14 + 1 = 15
    # Thread 6: shared[6] += shared[2]  → 18 + 3 = 21
    # Thread 7: shared[7] += shared[3]  → 22 + 6 = 28
    # 최종 결과:     [0, 1, 3, 6, 10, 15, 21, 28]

    offset = 1  # 시작 거리

    # log₂(TPB) = log₂(8) = 3번 반복
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        # Race Condition 방지를 위한 2단계 접근법:
        # 1단계: 읽기 (Read Phase)
        # 2단계: 쓰기 (Write Phase)
        #
        # 문제 상황: 여러 스레드가 동시에 같은 메모리 위치를 읽고 쓸 때
        # 발생할 수 있는 데이터 경합 상태를 방지해야 합니다.

        # 읽기 단계: 각 스레드가 필요한 값을 지역 변수에 저장
        current_val = shared[0]  # 기본값 (사용되지 않을 수 있음)
        if local_i >= offset and local_i < size:
            current_val = shared[local_i - offset]  # 실제 읽을 값

        barrier()  # 모든 읽기 완료 대기

        # 쓰기 단계: 읽은 값을 사용하여 누적 합 계산
        if local_i >= offset and local_i < size:
            shared[local_i] += current_val

        barrier()  # 모든 쓰기 완료 대기

        offset *= 2  # 다음 반복을 위해 거리 2배 증가

    # 3단계: 결과를 글로벌 메모리에 저장
    if global_i < size:
        output[global_i] = shared[local_i]


# ANCHOR_END: prefix_sum_simple

# ANCHOR: prefix_sum_complete
# Multi-Block Prefix Sum (다중 블록 Prefix Sum)
#
# 문제 상황:
# 실제 데이터는 단일 블록으로 처리하기에는 너무 클 수 있습니다.
# GPU의 블록 간에는 직접적인 통신 방법이 없으므로 새로운 접근법이 필요합니다.
#
# 해결 전략: 2단계 알고리즘 (Two-Phase Algorithm)
#
# Phase 1 (Local Phase): 각 블록 내에서 독립적으로 Prefix Sum 계산
# - Block 0: [0,1,2,3,4,5,6,7] → [0,1,3,6,10,15,21,28]
# - Block 1: [8,9,10,11,12,13,14] → [8,17,27,38,50,63,77]
#
# 문제: Block 1의 결과가 잘못됨! (Block 0의 합계를 반영하지 않음)
# 올바른 결과: Block 1은 [36,45,55,66,78,91,105]가 되어야 함
#
# Phase 2 (Global Phase): 이전 블록들의 합계를 현재 블록에 추가
# - Block 0의 마지막 값 (28)을 Block 1의 모든 원소에 더함
# - Block 1: [8,17,27,38,50,63,77] + 28 → [36,45,55,66,78,91,105]
#
# 핵심 도전 과제:
# 1. 블록 간 통신 방법 (Block-to-Block Communication)
# 2. 블록 간 동기화 (Cross-Block Synchronization)
# 3. 보조 메모리 관리 (Auxiliary Memory Management)

# Complete Case 구성 상수들
alias SIZE_2 = 15  # 더 큰 입력 배열 (다중 블록 필요)
alias BLOCKS_PER_GRID_2 = (2, 1)  # 2개 블록 사용
alias THREADS_PER_BLOCK_2 = (TPB, 1)
alias EXTENDED_SIZE = SIZE_2 + 2  # 원본 데이터 + 블록 합계 저장 공간
alias extended_layout = Layout.row_major(EXTENDED_SIZE)


# Phase 1 커널: 로컬 Prefix Sum 계산 및 블록 합계 저장
#
# 이 커널의 역할:
# 1. 각 블록 내에서 독립적으로 Prefix Sum 계산
# 2. 각 블록의 마지막 값(블록 합계)을 보조 메모리에 저장
# 3. Phase 2에서 사용할 블록 간 통신 데이터 준비
fn prefix_sum_local_phase[
    out_layout: Layout, in_layout: Layout
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    size: Int,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 20 lines)

    shared = tb[dtype]().row_major[TPB]().shared().alloc()

    # 데이터 로딩 단계
    #
    # 메모리 레이아웃 예시 (SIZE_2=15, TPB=8, BLOCKS=2):
    # Block 0: Thread 0~7이 a[0]~a[7] 로드
    # Block 1: Thread 0~6이 a[8]~a[14] 로드, Thread 7은 범위 초과로 로드하지 않음
    #
    # Block 0 shared memory: [0,1,2,3,4,5,6,7]
    # Block 1 shared memory: [8,9,10,11,12,13,14,uninitialized]
    #
    # 주의: Block 1의 Thread 7은 global_i=15 >= size이므로 데이터를 로드하지 않습니다.
    # 이는 안전합니다. 해당 스레드는 계산에 참여하지 않기 때문입니다.
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()  # 모든 스레드가 데이터 로딩 완료 대기

    # 로컬 Prefix Sum 계산 (Simple Case와 동일한 알고리즘)
    #
    # 트리 기반 병렬 리덕션을 사용하여 log(TPB) 반복으로 완료
    #
    # Block 0 계산 과정:
    # Iteration 1 (offset=1): [0,1,2,3,4,5,6,7] → [0,1,3,5,7,9,11,13]
    # Iteration 2 (offset=2): [0,1,3,5,7,9,11,13] → [0,1,3,6,10,14,18,22]
    # Iteration 3 (offset=4): [0,1,3,6,10,14,18,22] → [0,1,3,6,10,15,21,28]
    #
    # Block 1 계산 과정 (7개 유효 원소만):
    # 동일한 패턴으로 계산하여 [8,17,27,38,50,63,77,...]을 얻음
    offset = 1

    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val = shared[0]
        if local_i >= offset and local_i < TPB:
            current_val = shared[local_i - offset]  # 읽기

        barrier()  # 모든 스레드가 읽기 완료 대기

        if local_i >= offset and local_i < TPB:
            shared[local_i] += current_val  # 쓰기

        barrier()  # 모든 스레드가 쓰기 완료 대기

        offset *= 2

    # 로컬 결과를 출력 배열에 저장
    # Block 0: output[0~7] = [0,1,3,6,10,15,21,28]
    # Block 1: output[8~14] = [8,17,27,38,50,63,77]
    if global_i < size:
        output[global_i] = shared[local_i]

    # 핵심: 블록 합계를 보조 공간에 저장
    #
    # 블록 간 통신을 위한 핵심 메커니즘:
    # 각 블록의 마지막 스레드(TPB-1)가 해당 블록의 최종 합계를
    # 출력 배열의 확장된 영역에 저장합니다.
    #
    # 저장 위치: output[size + block_idx.x]
    # - Block 0: output[15 + 0] = output[15] = 28 (Block 0의 합계)
    # - Block 1: output[15 + 1] = output[16] = 77 (Block 1의 합계, 하지만 사용되지 않음)
    #
    # 메모리 레이아웃:
    # [0,1,3,6,10,15,21,28, 8,17,27,38,50,63,77, 28, 77]
    #  ←---- Block 0 ----→  ←--- Block 1 ----→   ↑   ↑
    #                                          Block sums
    if local_i == TPB - 1:
        output[size + block_idx.x] = shared[local_i]


# Phase 2 커널: 블록 합계 추가
#
# 이 커널의 역할:
# 1. 이전 블록들의 합계를 현재 블록의 모든 원소에 추가
# 2. 전역적으로 올바른 Prefix Sum 결과 생성
# 3. Block 0은 변경하지 않음 (이미 올바름)
fn prefix_sum_block_sum_phase[
    layout: Layout
](output: LayoutTensor[mut=False, dtype, layout], size: Int):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 3 lines)

    # 블록 합계 추가 로직
    #
    # Block 0 (block_idx.x = 0): 아무 작업 하지 않음 (이미 올바른 결과)
    # Block 1 (block_idx.x = 1): 이전 블록(Block 0)의 합계를 모든 원소에 추가
    #
    # 계산 과정:
    # 1. prev_block_sum = output[size + block_idx.x - 1]
    #    = output[15 + 1 - 1] = output[15] = 28
    # 2. Block 1의 각 원소에 28을 추가:
    #    Before: [8, 17, 27, 38, 50, 63, 77]
    #    After:  [36, 45, 55, 66, 78, 91, 105]
    #
    # 최종 결과: [0,1,3,6,10,15,21,28, 36,45,55,66,78,91,105]
    if block_idx.x > 0 and global_i < size:
        prev_block_sum = output[size + block_idx.x - 1]
        output[global_i] += prev_block_sum


# ANCHOR_END: prefix_sum_complete


def main():
    with DeviceContext() as ctx:
        # 실행 모드 결정: Simple vs Complete
        use_simple = argv()[1] == "--simple"
        size = SIZE if use_simple else SIZE_2
        num_blocks = (size + TPB - 1) // TPB  # 필요한 블록 수 계산

        # Complete Case에서 확장된 버퍼 크기 검증
        if not use_simple and num_blocks > EXTENDED_SIZE - SIZE_2:
            raise Error("Extended buffer too small for the number of blocks")

        # 메모리 할당
        # Simple Case: 원본 크기만 필요
        # Complete Case: 원본 크기 + 블록 합계 저장 공간 필요
        buffer_size = size if use_simple else EXTENDED_SIZE
        out = ctx.enqueue_create_buffer[dtype](buffer_size).enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)

        # 입력 데이터 초기화: [0, 1, 2, 3, 4, 5, ...]
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        a_tensor = LayoutTensor[mut=False, dtype, layout](a.unsafe_ptr())

        if use_simple:
            # Simple Case 실행: 단일 커널로 완료
            out_tensor = LayoutTensor[mut=False, dtype, layout](
                out.unsafe_ptr()
            )

            ctx.enqueue_function[prefix_sum_simple[layout]](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            # Complete Case 실행: 2단계 커널 실행
            var out_tensor = LayoutTensor[mut=False, dtype, extended_layout](
                out.unsafe_ptr()
            )

            # ANCHOR: prefix_sum_complete_block_level_sync
            # 새로운 개념: 블록 간 동기화 (Cross-Block Synchronization)
            #
            # GPU 동기화의 계층 구조:
            # 1. 스레드 간 동기화 (Intra-Block): barrier() 사용
            #    - 같은 블록 내 스레드들 간의 동기화
            #    - 공유 메모리 접근 시 Race Condition 방지
            #
            # 2. 블록 간 동기화 (Inter-Block): ctx.synchronize() 사용
            #    - 서로 다른 블록들 간의 동기화
            #    - 커널 실행 완료 대기
            #    - 호스트-디바이스 간 동기화
            #
            # 중요한 차이점:
            # - barrier(): 블록 내 스레드 동기화, GPU 내에서 실행
            # - ctx.synchronize(): 커널 간 동기화, 호스트에서 GPU 대기

            # Phase 1: 로컬 Prefix Sum 계산
            ctx.enqueue_function[
                prefix_sum_local_phase[extended_layout, extended_layout]
            ](
                out_tensor,
                a_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

            # 핵심: 블록 간 동기화
            #
            # 왜 ctx.synchronize()가 필요한가?
            # 1. Phase 1이 완전히 끝나야 Phase 2가 시작될 수 있음
            # 2. Phase 2는 Phase 1에서 저장한 블록 합계 데이터를 읽어야 함
            # 3. GPU의 비동기 실행 특성상 명시적 동기화 없이는 데이터 무결성 보장 불가
            #
            # 동기화 없이 발생할 수 있는 문제:
            # - Phase 2가 Phase 1 완료 전에 시작될 수 있음
            # - 블록 합계 데이터가 아직 메모리에 쓰여지지 않았을 수 있음
            # - 잘못된 결과나 정의되지 않은 동작 발생 가능
            ctx.synchronize()

            # Phase 2: 블록 합계 추가
            ctx.enqueue_function[prefix_sum_block_sum_phase[extended_layout]](
                out_tensor,
                size,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
            # ANCHOR_END: prefix_sum_complete_block_level_sync

        # 결과 검증을 위한 CPU 참조 구현
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)
        ctx.synchronize()  # GPU 작업 완료 대기

        # 순차적 Prefix Sum 계산 (검증용)
        with a.map_to_host() as a_host:
            expected[0] = a_host[0]  # 첫 번째 원소는 그대로
            for i in range(1, size):
                expected[i] = expected[i - 1] + a_host[i]  # 이전 합계 + 현재 값

        with out.map_to_host() as out_host:
            if not use_simple:
                print(
                    "Note: we print the extended buffer here, but we only need"
                    " to print the first `size` elements"
                )

            print("out:", out_host)  # GPU 결과 (확장된 버퍼 포함)
            print("expected:", expected)  # CPU 참조 결과

            # 정확성 검증: 원본 배열 크기만큼만 비교
            size = size if use_simple else SIZE_2
            for i in range(size):
                assert_equal(out_host[i], expected[i])
