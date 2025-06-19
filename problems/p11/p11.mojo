from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal

# ANCHOR: conv_1d_simple
# 1D 컨볼루션(1D Convolution) - Simple Case
#
# 컨볼루션(Convolution)의 핵심 개념:
# 컨볼루션은 신호 처리와 이미지 분석에서 사용되는 기본 연산으로,
# 두 시퀀스(배열)를 결합하여 세 번째 시퀀스를 생성하는 연산입니다.
#
# 수학적 표현: output[i] = Σ(j=0 to CONV-1) a[i+j] × b[j]
#
# 슬라이딩 윈도우(Sliding Window) 개념:
# 커널(kernel) b가 입력 배열 a 위를 슬라이딩하면서 각 위치에서
# 가중 합(weighted sum)을 계산하는 방식입니다.
#
# 예시: a = [0, 1, 2, 3, 4, 5], b = [0, 1, 2] (커널)
# - output[0] = a[0]×b[0] + a[1]×b[1] + a[2]×b[2] = 0×0 + 1×1 + 2×2 = 5
# - output[1] = a[1]×b[0] + a[2]×b[1] + a[3]×b[2] = 1×0 + 2×1 + 3×2 = 8
# - output[2] = a[2]×b[0] + a[3]×b[1] + a[4]×b[2] = 2×0 + 3×1 + 4×2 = 11
# - output[3] = a[3]×b[0] + a[4]×b[1] + a[5]×b[2] = 3×0 + 4×1 + 5×2 = 14
# - output[4] = a[4]×b[0] + a[5]×b[1] + 0×b[2] = 4×0 + 5×1 + 0×2 = 5 (경계 처리)
# - output[5] = a[5]×b[0] + 0×b[1] + 0×b[2] = 5×0 + 0×1 + 0×2 = 0 (경계 처리)

alias TPB = 8  # Threads Per Block - 블록당 스레드 수 (8개)
alias SIZE = 6  # 입력 배열 크기 (6개 요소)
alias CONV = 3  # 컨볼루션 커널 크기 (3개 요소)
alias BLOCKS_PER_GRID = (1, 1)  # 그리드당 블록 수 (1개 블록만 사용 - Simple Case)
alias THREADS_PER_BLOCK = (TPB, 1)  # 블록당 스레드 수 (8개 스레드)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)
alias in_layout = Layout.row_major(SIZE)  # 입력 배열용 레이아웃 (크기 6)
alias out_layout = Layout.row_major(SIZE)  # 출력 배열용 레이아웃 (크기 6)
alias conv_layout = Layout.row_major(CONV)  # 컨볼루션 커널용 레이아웃 (크기 3)


# GPU 커널 함수: Simple Case 1D 컨볼루션
#
# Simple Case의 특징:
# 1. 단일 블록 사용 (BLOCKS_PER_GRID = 1)
# 2. 모든 데이터가 하나의 블록 내에서 처리
# 3. 블록 경계 문제 없음 (데이터 크기 < 블록 크기)
# 4. 간단한 공유 메모리 관리
# 5. 기본적인 컨볼루션 개념 학습에 적합
fn conv_1d_simple[
    in_layout: Layout,  # 입력 배열 레이아웃 매개변수
    out_layout: Layout,  # 출력 배열 레이아웃 매개변수
    conv_layout: Layout,  # 컨볼루션 커널 레이아웃 매개변수
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x  # 글로벌 스레드 인덱스
    local_i = thread_idx.x  # 로컬 스레드 인덱스 (공유 메모리 접근용)

    # FILL ME IN (roughly 14 lines)

    # Simple Case에서는 전체 데이터가 하나의 블록에 들어가므로
    # 입력 배열과 커널 전체를 공유 메모리에 로드할 수 있습니다.
    #
    # 메모리 할당 크기:
    # - shared_a: SIZE (6개 요소) - 전체 입력 배열
    # - shared_b: CONV (3개 요소) - 전체 컨볼루션 커널
    shared_a = tb[dtype]().row_major[SIZE]().shared().alloc()  # 입력 배열용 공유 메모리
    shared_b = tb[dtype]().row_major[CONV]().shared().alloc()  # 커널용 공유 메모리

    # 1단계: 입력 배열을 공유 메모리로 로드
    # 각 스레드가 하나의 입력 요소를 담당하여 로드합니다.
    #
    # 로딩 패턴:
    # - Thread 0: a[0] → shared_a[0]
    # - Thread 1: a[1] → shared_a[1]
    # - Thread 2: a[2] → shared_a[2]
    # - Thread 3: a[3] → shared_a[3]
    # - Thread 4: a[4] → shared_a[4]
    # - Thread 5: a[5] → shared_a[5]
    # - Thread 6, 7: 유효하지 않은 인덱스이므로 로드하지 않음
    if global_i < SIZE:
        shared_a[local_i] = a[global_i]

    # 2단계: 컨볼루션 커널을 공유 메모리로 로드
    # 커널 크기가 작으므로 처음 몇 개 스레드만 로드를 담당합니다.
    #
    # 로딩 패턴:
    # - Thread 0: b[0] → shared_b[0]
    # - Thread 1: b[1] → shared_b[1]
    # - Thread 2: b[2] → shared_b[2]
    # - Thread 3~7: 커널 크기를 초과하므로 로드하지 않음
    if global_i < CONV:
        shared_b[local_i] = b[global_i]

    # 동기화 배리어: 모든 데이터 로딩 완료 대기
    # 모든 스레드가 입력 배열과 커널을 공유 메모리에 완전히 로드한 후
    # 컨볼루션 계산을 시작하도록 동기화합니다.
    barrier()

    # 3단계: 컨볼루션 계산 수행
    # 각 스레드가 하나의 출력 요소를 계산합니다.
    if global_i < SIZE:
        # 지역 합계 변수 초기화
        # output.element_type을 사용하여 출력 텐서와 동일한 타입으로 초기화
        var local_sum: output.element_type = 0

        # @parameter 데코레이터는 컴파일 타임에 루프를 "언롤(unroll)"합니다.
        # 이는 루프를 실제 반복문이 아닌 연속된 코드로 변환하는 최적화 기법입니다.
        #
        # 언롤링 전 (런타임 루프):
        # for j in range(CONV):  // 런타임에 3번 반복
        #     local_sum += shared_a[local_i + j] * shared_b[j]
        #
        # 언롤링 후 (컴파일 타임 확장):
        # local_sum += shared_a[local_i + 0] * shared_b[0]  // j=0
        # local_sum += shared_a[local_i + 1] * shared_b[1]  // j=1
        # local_sum += shared_a[local_i + 2] * shared_b[2]  // j=2
        #
        # @parameter 사용 조건:
        # 1. 루프 시퀀스가 컴파일 타임에 결정되어야 함 (CONV는 alias 상수)
        # 2. 인덕션 변수(j)가 유효한 매개변수 표현식이어야 함
        #
        # 장점:
        # 1. 루프 오버헤드 제거 (조건 검사, 점프 명령어 없음)
        # 2. 더 나은 컴파일러 최적화 기회
        # 3. 예측 가능한 성능
        # 4. 작은 고정 크기 루프에 매우 효과적
        @parameter
        for j in range(CONV):
            # 경계 검사: 입력 배열 범위를 벗어나지 않도록 확인
            # local_i + j < SIZE 조건으로 유효한 데이터만 처리
            #
            # 컨볼루션 계산:
            # shared_a[local_i + j]: 현재 위치에서 j만큼 떨어진 입력 값
            # shared_b[j]: 커널의 j번째 가중치
            # 두 값의 곱을 누적 합산
            if local_i + j < SIZE:
                local_sum += shared_a[local_i + j] * shared_b[j]

        # 계산된 결과를 출력 배열에 저장
        output[global_i] = local_sum


# ANCHOR_END: conv_1d_simple

# ANCHOR: conv_1d_block_boundary
# Block Boundary Case - 블록 경계 처리
#
# Block Boundary Case의 필요성:
# 실제 GPU 프로그래밍에서는 데이터 크기가 단일 블록으로 처리하기에는
# 너무 클 수 있습니다. 이 경우 여러 블록을 사용해야 하며,
# 블록 간 경계에서 데이터 공유 문제가 발생합니다.
#
# 문제 상황:
# - 컨볼루션은 슬라이딩 윈도우 연산이므로 인접 데이터가 필요
# - 블록 경계에서 다음 블록의 데이터에 접근해야 함
# - 각 블록은 독립적인 공유 메모리를 가지므로 직접 접근 불가
#
# 해결 방법:
# - 확장된 공유 메모리 할당 (TPB + CONV - 1)
# - 블록 경계 데이터를 추가로 로드
# - 적절한 경계 조건 처리

# Block Boundary Case 구성 상수들
alias SIZE_2 = 15  # 입력 배열 크기 (15개 요소 - 더 큰 데이터)
alias CONV_2 = 4  # 컨볼루션 커널 크기 (4개 요소 - 더 큰 커널)
alias BLOCKS_PER_GRID_2 = (2, 1)  # 그리드당 블록 수 (2개 블록 사용)
alias THREADS_PER_BLOCK_2 = (TPB, 1)  # 블록당 스레드 수 (8개 스레드)
alias in_2_layout = Layout.row_major(SIZE_2)  # 입력 배열용 레이아웃 (크기 15)
alias out_2_layout = Layout.row_major(SIZE_2)  # 출력 배열용 레이아웃 (크기 15)
alias conv_2_layout = Layout.row_major(CONV_2)  # 컨볼루션 커널용 레이아웃 (크기 4)


# GPU 커널 함수: Block Boundary Case 1D 컨볼루션
#
# Block Boundary Case의 특징:
# 1. 다중 블록 사용 (BLOCKS_PER_GRID_2 = 2)
# 2. 블록 간 데이터 의존성 존재
# 3. 확장된 공유 메모리 필요 (TPB + CONV_2 - 1)
# 4. 복잡한 경계 조건 처리
# 5. 실제 GPU 프로그래밍 시나리오에 가까움
#
# 메모리 레이아웃 예시:
# 전체 데이터: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# Block 0 담당: [0, 1, 2, 3, 4, 5, 6, 7] + 경계 데이터 [8, 9, 10]
# Block 1 담당: [8, 9, 10, 11, 12, 13, 14] + 경계 데이터 (패딩)
fn conv_1d_block_boundary[
    in_layout: Layout,  # 입력 배열 레이아웃 매개변수
    out_layout: Layout,  # 출력 배열 레이아웃 매개변수
    conv_layout: Layout,  # 컨볼루션 커널 레이아웃 매개변수
    dtype: DType,  # 데이터 타입 매개변수 (추가됨)
](
    output: LayoutTensor[mut=False, dtype, out_layout],
    a: LayoutTensor[mut=False, dtype, in_layout],
    b: LayoutTensor[mut=False, dtype, conv_layout],
):
    # 스레드 인덱스 계산
    global_i = block_dim.x * block_idx.x + thread_idx.x  # 글로벌 스레드 인덱스
    local_i = thread_idx.x  # 로컬 스레드 인덱스

    # FILL ME IN (roughly 18 lines)

    # 공유 메모리 할당 - Block Boundary Case
    #
    # 핵심 차이점: 확장된 공유 메모리 크기
    # Simple Case: SIZE (6개) vs Block Boundary: TPB + CONV_2 - 1 (8 + 4 - 1 = 11개)
    #
    # 크기 계산 이유:
    # - TPB: 현재 블록의 기본 데이터 (8개)
    # - CONV_2 - 1: 다음 블록에서 필요한 추가 데이터 (3개)
    # - 총 11개 요소로 블록 경계를 넘나드는 컨볼루션 윈도우 처리 가능
    #
    # 메모리 구조:
    # shared_a: [블록 데이터 8개 | 경계 데이터 3개] = 총 11개
    shared_a = (
        tb[dtype]().row_major[TPB + CONV_2 - 1]().shared().alloc()
    )  # 확장된 입력용 공유 메모리
    shared_b = tb[dtype]().row_major[CONV_2]().shared().alloc()  # 커널용 공유 메모리

    # 1단계: 기본 블록 데이터 로드
    # 각 스레드가 자신의 담당 데이터를 로드합니다.
    #
    # Block 0: Thread 0~7이 a[0]~a[7] 로드
    # Block 1: Thread 0~7이 a[8]~a[14] 로드 (Thread 7은 a[15] 접근 시도하지만 경계 검사로 방지)
    if global_i < SIZE_2:
        shared_a[local_i] = a[global_i]

    # 2단계: 블록 경계 데이터 로드 (핵심 차이점!)
    #
    # 블록 경계 처리의 핵심:
    # 컨볼루션 윈도우가 블록 경계를 넘나들 때 필요한 추가 데이터를 로드합니다.
    #
    # 로드 조건: local_i < CONV_2 - 1 (처음 3개 스레드만 참여)
    # - Thread 0, 1, 2만 경계 데이터 로드 담당
    # - Thread 3~7은 경계 데이터 로드에 참여하지 않음 (효율성)
    #
    # 메모리 접근 패턴:
    # Block 0: Thread 0~2가 a[8], a[9], a[10] 로드 → shared_a[8], shared_a[9], shared_a[10]
    # Block 1: Thread 0~2가 a[16], a[17], a[18] 접근 시도 (범위 초과로 0으로 패딩)
    if local_i < CONV_2 - 1:
        # 다음 블록 영역의 데이터 인덱스 계산
        next_idx = global_i + TPB  # 현재 위치에서 TPB(8)만큼 앞선 위치

        if next_idx < SIZE_2:
            # 유효한 데이터 범위 내: 실제 데이터 로드
            shared_a[TPB + local_i] = a[next_idx]
        else:
            # 데이터 범위 초과: 0으로 패딩 (경계 조건 처리)
            # 이는 컨볼루션에서 일반적인 제로 패딩(zero padding) 기법입니다.
            shared_a[TPB + local_i] = 0

    # 3단계: 컨볼루션 커널 로드
    # Simple Case와 동일하지만 커널 크기가 CONV_2(4)로 더 큽니다.
    if local_i < CONV_2:
        shared_b[local_i] = b[local_i]

    # 동기화 배리어: 모든 데이터 로딩 완료 대기
    # 기본 데이터, 경계 데이터, 커널 데이터 모두 로드 완료 후 계산 시작
    barrier()

    # 4단계: 컨볼루션 계산 수행
    # Simple Case와 유사하지만 더 큰 커널(CONV_2=4)과 경계 조건 처리가 다릅니다.
    if global_i < SIZE_2:
        # 지역 합계 변수 초기화
        var local_sum: output.element_type = 0

        # @parameter 데코레이터로 컴파일 타임 루프 언롤링
        # CONV_2(4) 크기의 루프가 4개의 연속된 명령어로 확장됩니다.
        @parameter
        for j in range(CONV_2):
            # 중요한 차이점: 경계 조건 검사
            # Simple Case: if local_i + j < SIZE (공유 메모리 경계 검사)
            # Block Boundary: if global_i + j < SIZE_2 (전체 데이터 경계 검사)
            #
            # 이유:
            # - global_i + j < SIZE_2: 전체 입력 배열의 유효한 범위 내에서만 계산
            # - 수학적으로 올바른 컨볼루션 연산 보장
            # - 확장된 공유 메모리의 제로 패딩 영역 활용
            if global_i + j < SIZE_2:
                # 컨볼루션 계산:
                # shared_a[local_i + j]: 확장된 공유 메모리에서 데이터 접근
                # shared_b[j]: 커널의 j번째 가중치
                local_sum += shared_a[local_i + j] * shared_b[j]

        # 계산된 결과를 출력 배열에 저장
        output[global_i] = local_sum


# ANCHOR_END: conv_1d_block_boundary


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    with DeviceContext() as ctx:
        # 명령행 인수에 따라 실행할 케이스 결정
        # argv()[1]: "--simple" 또는 "--block-boundary"
        #
        # 동적 구성 설정:
        # Simple Case: SIZE=6, CONV=3, 1개 블록
        # Block Boundary Case: SIZE_2=15, CONV_2=4, 2개 블록
        size = SIZE_2 if argv()[1] == "--block-boundary" else SIZE
        conv = CONV_2 if argv()[1] == "--block-boundary" else CONV

        # GPU 메모리 버퍼 생성 및 초기화
        out = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)  # 출력 배열 버퍼
        a = ctx.enqueue_create_buffer[dtype](size).enqueue_fill(0)  # 입력 배열 버퍼
        b = ctx.enqueue_create_buffer[dtype](conv).enqueue_fill(0)  # 커널 배열 버퍼

        # 입력 데이터 초기화: [0, 1, 2, 3, 4, 5, ...] 순차적 값
        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        # 커널 데이터 초기화: [0, 1, 2, ...] 순차적 값
        with b.map_to_host() as b_host:
            for i in range(conv):
                b_host[i] = i

        # 케이스별 GPU 커널 실행
        if argv()[1] == "--simple":
            # Simple Case 실행
            #
            # 특징:
            # - 단일 블록 처리
            # - 작은 데이터 크기 (SIZE=6, CONV=3)
            # - 기본적인 컨볼루션 개념 학습
            # - 블록 경계 문제 없음
            var out_tensor = LayoutTensor[mut=False, dtype, out_layout](
                out.unsafe_ptr()
            )
            var a_tensor = LayoutTensor[mut=False, dtype, in_layout](
                a.unsafe_ptr()
            )
            var b_tensor = LayoutTensor[mut=False, dtype, conv_layout](
                b.unsafe_ptr()
            )
            ctx.enqueue_function[
                conv_1d_simple[in_layout, out_layout, conv_layout]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,  # (1, 1) - 단일 블록
                block_dim=THREADS_PER_BLOCK,  # (8, 1) - 8개 스레드
            )
        elif argv()[1] == "--block-boundary":
            # Block Boundary Case 실행
            #
            # 특징:
            # - 다중 블록 처리 (2개 블록)
            # - 큰 데이터 크기 (SIZE_2=15, CONV_2=4)
            # - 블록 경계 데이터 공유 처리
            # - 실제 GPU 프로그래밍 시나리오
            var out_tensor = LayoutTensor[mut=False, dtype, out_2_layout](
                out.unsafe_ptr()
            )
            var a_tensor = LayoutTensor[mut=False, dtype, in_2_layout](
                a.unsafe_ptr()
            )
            var b_tensor = LayoutTensor[mut=False, dtype, conv_2_layout](
                b.unsafe_ptr()
            )
            ctx.enqueue_function[
                conv_1d_block_boundary[
                    in_2_layout, out_2_layout, conv_2_layout, dtype
                ]
            ](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID_2,  # (2, 1) - 2개 블록
                block_dim=THREADS_PER_BLOCK_2,  # (8, 1) - 8개 스레드
            )
        else:
            raise Error("Invalid argument")

        # GPU 작업 완료 대기
        ctx.synchronize()

        # CPU에서 기대값 계산 (검증용)
        # GPU 결과와 비교하기 위해 CPU에서 동일한 컨볼루션 연산을 수행합니다
        expected = ctx.enqueue_create_host_buffer[dtype](size).enqueue_fill(0)

        # 순차적 컨볼루션 계산 (CPU 버전)
        # 이는 GPU 구현의 정확성을 검증하기 위한 참조 구현입니다
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(size):
                for j in range(conv):
                    # 경계 조건: i + j < size인 경우만 계산
                    # 이는 GPU 구현의 경계 검사와 동일한 로직입니다
                    if i + j < size:
                        expected[i] += a_host[i + j] * b_host[j]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(size):
                assert_equal(out_host[i], expected[i])
