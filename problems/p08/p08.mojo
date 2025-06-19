from memory import UnsafePointer, stack_allocation  # 메모리 포인터와 스택 할당을 위한 클래스들
from gpu import (
    thread_idx,
    block_idx,
    block_dim,
    barrier,
)  # GPU 스레드/블록 인덱스, 차원 정보, 동기화 배리어
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from gpu.memory import AddressSpace  # GPU 메모리 주소 공간 지정을 위한 열거형
from sys import sizeof  # 데이터 타입의 크기를 구하기 위한 함수
from testing import assert_equal

# ANCHOR: add_10_shared
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias TPB = 4  # Threads Per Block - 블록당 스레드 수 (4개)
alias SIZE = 8  # 1D 배열의 크기 (8개 요소)
alias BLOCKS_PER_GRID = (2, 1)  # 그리드당 블록 수 (2개 블록 사용)
alias THREADS_PER_BLOCK = (TPB, 1)  # 블록당 스레드 수 (4개 스레드) - 데이터보다 적음!
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)


# GPU 커널 함수: 공유 메모리를 사용한 GPU 병렬 처리
# 이 함수는 블록 내 스레드들이 공유 메모리를 통해 데이터를 공유하고 동기화하는 패턴입니다
# 핵심: 공유 메모리는 같은 블록 내 스레드들만 접근 가능한 고속 메모리입니다
fn add_10_shared(
    output: UnsafePointer[Scalar[dtype]],  # 출력 결과를 저장할 메모리 포인터
    a: UnsafePointer[Scalar[dtype]],  # 입력 1D 배열이 저장된 메모리 포인터
    size: Int,  # 1D 배열의 크기 (8개 요소)
):
    # 공유 메모리(Shared Memory) 할당
    # 공유 메모리는 같은 블록 내 모든 스레드가 공유하는 고속 메모리입니다
    # 특징:
    # 1. 글로벌 메모리보다 훨씬 빠름 (레이턴시 ~100배 낮음)
    # 2. 같은 블록 내 스레드들만 접근 가능
    # 3. 블록당 제한된 크기 (보통 48KB 정도)
    # 4. 명시적 동기화 필요 (barrier 사용)
    shared = stack_allocation[
        TPB,  # 공유 메모리 크기: 블록당 스레드 수만큼 (4개 요소)
        Scalar[dtype],  # 데이터 타입: float32
        address_space = AddressSpace.SHARED,  # 메모리 주소 공간: 공유 메모리 지정
    ]()

    # 글로벌 스레드 인덱스 계산 (전체 데이터에서의 위치)
    # 공식: global_i = block_dim.x * block_idx.x + thread_idx.x
    # 예시 (블록당 4개 스레드, 2개 블록):
    # 블록 0: 스레드 0,1,2,3 → 글로벌 인덱스 0,1,2,3
    # 블록 1: 스레드 0,1,2,3 → 글로벌 인덱스 4,5,6,7
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # 로컬 스레드 인덱스 (블록 내에서의 위치)
    # 공유 메모리 접근 시 사용하는 인덱스
    # 값 범위: 0 ~ (TPB-1), 즉 0,1,2,3
    local_i = thread_idx.x

    # 글로벌 메모리에서 공유 메모리로 데이터 로드
    # 각 스레드가 자신이 담당하는 하나의 요소를 공유 메모리에 복사합니다
    # 목적: 이후 계산에서 빠른 공유 메모리를 사용하기 위함
    if global_i < size:
        # 글로벌 메모리 a[global_i] → 공유 메모리 shared[local_i]
        # 예: 블록 0에서 스레드 2는 a[2] → shared[2]로 복사
        # 예: 블록 1에서 스레드 1은 a[5] → shared[1]로 복사
        shared[local_i] = a[global_i]

    # 동기화 배리어(Synchronization Barrier)
    # barrier() 함수는 CUDA의 __syncthreads()와 동일한 기능을 수행합니다
    #
    # 동작 원리:
    # 1. 블록 내 모든 스레드가 이 지점에 도달할 때까지 대기
    # 2. 모든 스레드가 도착하면 동시에 다음 단계로 진행
    # 3. 메모리 일관성 보장: 배리어 이전의 메모리 연산이 모든 스레드에게 보임
    #
    # 필요한 이유:
    # - 스레드들이 서로 다른 속도로 실행될 수 있음
    # - 일부 스레드가 공유 메모리 쓰기를 완료하기 전에 다른 스레드가 읽으면 오류
    # - 데이터 레이스(race condition) 방지
    #
    # 예시 시나리오 (배리어 없이):
    # 스레드 0: shared[0] = a[0] (완료) → shared[1] 읽기 시도 (아직 쓰여지지 않음!)
    # 스레드 1: shared[1] = a[1] (진행 중...)
    barrier()

    # 공유 메모리에서 데이터를 읽어 계산 수행
    # 모든 스레드가 공유 메모리 로딩을 완료한 후 안전하게 계산을 수행합니다
    # 이 예제에서는 단순히 각 스레드가 자신의 데이터에만 접근하지만,
    # 실제 응용에서는 다른 스레드의 데이터에도 접근할 수 있습니다
    if global_i < size:
        # 공유 메모리에서 데이터를 읽어 10을 더한 후 글로벌 메모리에 저장
        # shared[local_i]: 공유 메모리에서 빠르게 읽기
        # output[global_i]: 결과를 글로벌 메모리에 저장
        output[global_i] = shared[local_i] + 10.0


# ANCHOR_END: add_10_shared


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # 이 컨텍스트는 GPU 메모리 할당, 데이터 복사, 커널 실행 등을 담당합니다
    with DeviceContext() as ctx:
        # 1D 배열을 위한 GPU 메모리 버퍼들을 생성합니다
        # SIZE = 8개의 요소를 가진 1차원 배열
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 출력 결과 버퍼
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
            1
        )  # 입력 배열 버퍼 (모든 요소 1.0으로 초기화)

        # GPU 커널 함수를 실행합니다
        # 공유 메모리를 사용한 다중 블록 구성
        #
        # 블록 구성 분석:
        # - 총 데이터: 8개 요소 [0,1,2,3,4,5,6,7]
        # - 블록 수: 2개 (BLOCKS_PER_GRID = (2,1))
        # - 블록당 스레드: 4개 (THREADS_PER_BLOCK = (4,1))
        # - 총 스레드: 2 × 4 = 8개 (데이터와 정확히 일치!)
        #
        # 공유 메모리 사용 패턴:
        # 블록 0: 스레드 0,1,2,3 → 데이터 인덱스 0,1,2,3 처리
        #         공유 메모리 shared[0:4]에 a[0:4] 로드 후 계산
        # 블록 1: 스레드 0,1,2,3 → 데이터 인덱스 4,5,6,7 처리
        #         공유 메모리 shared[0:4]에 a[4:8] 로드 후 계산
        #
        # 메모리 계층 구조:
        # 1. 글로벌 메모리 (느림): a[0:8], output[0:8]
        # 2. 공유 메모리 (빠름): 각 블록마다 shared[0:4]
        # 3. 레지스터 (가장 빠름): 스레드별 local_i, global_i
        ctx.enqueue_function[add_10_shared](
            out.unsafe_ptr(),  # 출력 버퍼의 메모리 포인터
            a.unsafe_ptr(),  # 입력 배열의 메모리 포인터
            SIZE,  # 배열 크기 (8)
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (2개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (4개 스레드)
        )

        # 기대값을 저장할 호스트 메모리 버퍼를 생성합니다
        # 모든 요소가 11.0 (1.0 + 10.0)이 되어야 합니다
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(11)

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        ctx.synchronize()

        # GPU에서 계산된 결과를 호스트 메모리로 매핑하여 확인합니다
        with out.map_to_host() as out_host:
            print(
                "out:", out_host
            )  # GPU 결과: [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]
            print(
                "expected:", expected
            )  # 기대값: [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]

            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
