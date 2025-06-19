from memory import UnsafePointer, stack_allocation
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from sys import sizeof
from testing import assert_equal

# ANCHOR: dot_product
alias TPB = 8
alias SIZE = 8
alias BLOCKS_PER_GRID = (1, 1)
alias THREADS_PER_BLOCK = (SIZE, 1)
alias dtype = DType.float32


# GPU 커널 함수: 내적(Dot Product) 연산
#
# 내적(Dot Product)의 핵심 개념:
# 내적은 두 벡터의 대응하는 원소들을 곱한 후 모든 결과를 더하는 연산입니다.
# 수학적 표현: a · b = a₀×b₀ + a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ
#
# 예시: 벡터 a = [0, 1, 2, 3, 4, 5, 6, 7], 벡터 b = [0, 1, 2, 3, 4, 5, 6, 7]
# - a[0] × b[0] = 0 × 0 = 0
# - a[1] × b[1] = 1 × 1 = 1
# - a[2] × b[2] = 2 × 2 = 4
# - a[3] × b[3] = 3 × 3 = 9
# - a[4] × b[4] = 4 × 4 = 16
# - a[5] × b[5] = 5 × 5 = 25
# - a[6] × b[6] = 6 × 6 = 36
# - a[7] × b[7] = 7 × 7 = 49
# 최종 결과: 0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 = 140
#
# 병렬 처리의 핵심 도전: 리덕션(Reduction) 문제
# 내적 계산은 두 단계로 나뉩니다:
# 1. 원소별 곱셈 (Element-wise Multiplication) - 병렬 처리 가능
# 2. 모든 곱셈 결과의 합산 (Sum Reduction) - 병렬 처리 어려움
#
# 트리 기반 리덕션(Tree-based Reduction) 알고리즘:
# 순차적 합산 대신 이진 트리 구조로 병렬 합산을 수행합니다.
#
# 예시 (8개 원소):
# 초기값: [0, 1, 4, 9, 16, 25, 36, 49]
#
# 1단계 (stride=4): 인접한 4칸 간격으로 합산
# - shared[0] += shared[4] → 0 + 16 = 16
# - shared[1] += shared[5] → 1 + 25 = 26
# - shared[2] += shared[6] → 4 + 36 = 40
# - shared[3] += shared[7] → 9 + 49 = 58
# 결과: [16, 26, 40, 58, 16, 25, 36, 49]
#
# 2단계 (stride=2): 인접한 2칸 간격으로 합산
# - shared[0] += shared[2] → 16 + 40 = 56
# - shared[1] += shared[3] → 26 + 58 = 84
# 결과: [56, 84, 40, 58, 16, 25, 36, 49]
#
# 3단계 (stride=1): 인접한 1칸 간격으로 합산
# - shared[0] += shared[1] → 56 + 84 = 140
# 최종 결과: [140, 84, 40, 58, 16, 25, 36, 49]
#
# 효율적인 구현 전략:
# 1. 공유 메모리 사용: 빠른 데이터 접근과 스레드 간 협력
# 2. 2 Global Reads per Thread: 각 스레드가 a[i]와 b[i]를 한 번씩 읽기
# 3. 1 Global Write per Block: 블록당 하나의 최종 결과만 출력
# 4. 트리 리덕션: log₂(n) 단계로 효율적인 병렬 합산
fn dot_product(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    # FILL ME IN (roughly 13 lines)

    # 공유 메모리(Shared Memory) 할당
    # 트리 리덕션을 위한 임시 저장 공간으로 사용됩니다
    #
    # 공유 메모리를 사용하는 이유:
    # 1. 빠른 접근 속도: 글로벌 메모리보다 훨씬 빠름
    # 2. 스레드 간 협력: 블록 내 모든 스레드가 공유 데이터에 접근 가능
    # 3. 리덕션 최적화: 중간 계산 결과를 효율적으로 저장하고 조합
    # 4. 메모리 대역폭 절약: 글로벌 메모리 접근 횟수 최소화
    shared = stack_allocation[
        TPB, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()

    # 스레드 인덱스 계산
    # 글로벌 스레드 인덱스: 전체 데이터에서의 위치
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # 로컬 스레드 인덱스: 블록 내에서의 위치 (공유 메모리 인덱스로 사용)
    local_i = thread_idx.x

    # 1단계: 원소별 곱셈 및 공유 메모리 로드 (Element-wise Multiplication)
    # 각 스레드가 대응하는 원소들을 곱하고 결과를 공유 메모리에 저장합니다.
    #
    # 병렬 처리 패턴:
    # - Thread 0: a[0] × b[0] → shared[0]
    # - Thread 1: a[1] × b[1] → shared[1]
    # - Thread 2: a[2] × b[2] → shared[2]
    # - ... 모든 스레드가 동시에 실행
    #
    # 경계 검사: global_i < size로 유효한 데이터 범위인지 확인
    # 스레드 수와 데이터 수가 다를 수 있으므로 안전성 확보
    if global_i < size:
        shared[local_i] = a[global_i] * b[global_i]

    # 동기화 배리어(Synchronization Barrier)
    # 모든 스레드가 원소별 곱셈을 완료할 때까지 기다립니다.
    #
    # 배리어가 필요한 이유:
    # 1. 데이터 일관성: 모든 곱셈 결과가 공유 메모리에 완전히 저장된 후 리덕션 시작
    # 2. 경쟁 조건 방지: 일부 스레드가 아직 쓰기 중인 데이터를 다른 스레드가 읽는 것 방지
    # 3. 단계별 동기화: 리덕션의 각 단계마다 모든 스레드가 동기화되어야 함
    barrier()

    # 2단계: 트리 기반 리덕션(Tree-based Reduction) 수행
    # 이진 트리 구조로 병렬 합산을 수행하여 최종 내적 결과를 계산합니다.
    #
    # 리덕션 알고리즘 설명:
    # - stride: 현재 단계에서 합산할 원소들 간의 거리
    # - 각 단계마다 stride가 절반으로 줄어듦 (TPB/2 → TPB/4 → TPB/8 → ... → 1)
    # - 활성 스레드 수도 각 단계마다 절반으로 줄어듦
    # - log₂(TPB) 단계 후 shared[0]에 최종 결과 저장

    # 초기 stride 설정: 전체 크기의 절반부터 시작
    stride = TPB // 2  # stride = 8 // 2 = 4

    # 트리 리덕션 루프: stride가 0이 될 때까지 반복
    while stride > 0:
        # 활성 스레드 선별: local_i < stride인 스레드만 참여
        # 각 단계마다 절반의 스레드만 활성화되어 효율적인 병렬 처리
        #
        # 예시 (TPB=8):
        # - stride=4: Thread 0,1,2,3 활성 (4개 스레드)
        # - stride=2: Thread 0,1 활성 (2개 스레드)
        # - stride=1: Thread 0 활성 (1개 스레드)
        if local_i < stride:
            # 쌍별 합산(Pairwise Addition) 수행
            # 현재 위치의 값과 stride만큼 떨어진 위치의 값을 합산
            #
            # 메모리 접근 패턴:
            # - shared[local_i]: 현재 스레드가 담당하는 위치
            # - shared[local_i + stride]: stride만큼 떨어진 위치
            #
            # 예시 (stride=4일 때):
            # - Thread 0: shared[0] += shared[4] (0번째와 4번째 합산)
            # - Thread 1: shared[1] += shared[5] (1번째와 5번째 합산)
            # - Thread 2: shared[2] += shared[6] (2번째와 6번째 합산)
            # - Thread 3: shared[3] += shared[7] (3번째와 7번째 합산)
            shared[local_i] += shared[local_i + stride]

        # 단계별 동기화 배리어
        # 현재 단계의 모든 합산이 완료될 때까지 기다립니다.
        # 다음 단계로 진행하기 전에 모든 스레드가 동기화되어야 합니다.
        barrier()

        # stride 절반으로 축소: 다음 단계 준비
        # 트리의 다음 레벨로 이동 (더 적은 수의 원소들을 합산)
        stride //= 2  # stride = stride // 2 (정수 나눗셈)

    # 3단계: 최종 결과 출력
    # 트리 리덕션이 완료되면 shared[0]에 최종 내적 결과가 저장됩니다.
    # Thread 0만이 최종 결과를 글로벌 메모리에 쓰기를 담당합니다.
    #
    # 왜 Thread 0만 쓰기를 하는가?
    # 1. 중복 쓰기 방지: 여러 스레드가 동시에 쓰면 비효율적
    # 2. 메모리 일관성: 단일 스레드 쓰기로 데이터 무결성 보장
    # 3. 효율성: 1 Global Write per Block 달성
    if local_i == 0:
        output[0] = shared[0]  # 최종 내적 결과를 출력 버퍼에 저장


# ANCHOR_END: dot_product


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    with DeviceContext() as ctx:
        # Dot Product 결과는 단일 스칼라 값
        # 이전 문제들과 달리 출력이 벡터가 아닌 단일 값입니다
        # 내적 연산의 특성: 두 벡터 → 하나의 숫자
        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(
            0
        )  # 출력 결과 버퍼 (크기 1, 내적 결과)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
            0
        )  # 첫 번째 벡터 버퍼 (크기 SIZE)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
            0
        )  # 두 번째 벡터 버퍼 (크기 SIZE)

        # 두 벡터 동시 초기화
        # 내적 연산을 위해 두 개의 입력 벡터가 필요합니다
        # with 문에서 여러 버퍼를 동시에 매핑하는 효율적인 패턴
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i  # 첫 번째 벡터: [0, 1, 2, 3, 4, 5, 6, 7]
                b_host[i] = i  # 두 번째 벡터: [0, 1, 2, 3, 4, 5, 6, 7]

        # GPU 커널 함수 실행
        # 리덕션 연산의 실행 구성
        # - 모든 데이터가 하나의 블록에서 처리되어야 함 (리덕션 특성)
        # - 블록 크기 = 데이터 크기 (THREADS_PER_BLOCK = SIZE)
        # - 트리 리덕션을 위해 스레드 수가 2의 거듭제곱이어야 효율적
        ctx.enqueue_function[dot_product](
            out.unsafe_ptr(),  # 출력 포인터 (내적 결과 저장)
            a.unsafe_ptr(),  # 첫 번째 벡터 포인터
            b.unsafe_ptr(),  # 두 번째 벡터 포인터
            SIZE,  # 벡터 크기
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (1개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (8개 스레드)
        )

        # 내적 결과는 단일 값이므로 크기가 1인 버퍼 생성
        expected = ctx.enqueue_create_host_buffer[dtype](1).enqueue_fill(0)

        ctx.synchronize()

        # GPU의 트리 리덕션 결과와 비교하기 위한 순차적 내적 계산
        # 두 벡터를 동시에 매핑하여 효율적인 검증 수행
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            # 순차적 내적 계산: 각 원소 쌍을 곱하고 누적 합산
            # 리덕션 검증의 핵심: GPU 병렬 결과 vs CPU 순차 결과 비교
            for i in range(SIZE):
                # expected[0] += a_host[i] * b_host[i]
                # 각 단계별 계산 (디버깅 및 학습용):
                # i=0: expected[0] += 0 * 0 = 0 (누적: 0)
                # i=1: expected[0] += 1 * 1 = 1 (누적: 1)
                # i=2: expected[0] += 2 * 2 = 4 (누적: 5)
                # i=3: expected[0] += 3 * 3 = 9 (누적: 14)
                # i=4: expected[0] += 4 * 4 = 16 (누적: 30)
                # i=5: expected[0] += 5 * 5 = 25 (누적: 55)
                # i=6: expected[0] += 6 * 6 = 36 (누적: 91)
                # i=7: expected[0] += 7 * 7 = 49 (누적: 140)
                expected[0] += a_host[i] * b_host[i]

        with out.map_to_host() as out_host:
            print("out:", out_host)  # GPU 내적 결과 (트리 리덕션)
            print("expected:", expected)  # CPU 내적 결과 (순차적 계산)
            assert_equal(out_host[0], expected[0])
