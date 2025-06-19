from memory import UnsafePointer  # 메모리 포인터를 다루기 위한 클래스
from gpu import thread_idx, block_idx, block_dim  # GPU 스레드/블록 인덱스 및 차원 정보
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from testing import assert_equal

# ANCHOR: add_10_blocks
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias SIZE = 9  # 1D 배열의 크기 (9개 요소)
alias BLOCKS_PER_GRID = (3, 1)  # 그리드당 블록 수 (3개 블록 사용)
alias THREADS_PER_BLOCK = (4, 1)  # 블록당 스레드 수 (4개 스레드) - 데이터보다 적음!
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)


# GPU 커널 함수: 다중 블록을 사용한 1D 배열 처리
# 이 함수는 데이터 크기가 단일 블록의 스레드 수보다 클 때 사용하는 패턴입니다
# 핵심: 여러 블록이 협력하여 큰 데이터를 처리합니다
fn add_10_blocks(
    output: UnsafePointer[Scalar[dtype]],  # 출력 결과를 저장할 메모리 포인터
    a: UnsafePointer[Scalar[dtype]],  # 입력 1D 배열이 저장된 메모리 포인터
    size: Int,  # 1D 배열의 크기 (9개 요소)
):
    # 🔑 핵심 개념: 글로벌 스레드 인덱스 계산
    # 각 스레드가 처리할 데이터의 전역 위치를 계산합니다
    # 공식: i = block_dim.x * block_idx.x + thread_idx.x
    #
    # 예시 (블록당 4개 스레드, 3개 블록):
    # 블록 0: 스레드 0,1,2,3 → 글로벌 인덱스 0,1,2,3
    # 블록 1: 스레드 0,1,2,3 → 글로벌 인덱스 4,5,6,7
    # 블록 2: 스레드 0,1,2,3 → 글로벌 인덱스 8,9,10,11
    #
    # 계산 과정:
    # - block_dim.x: 블록당 스레드 수 (4)
    # - block_idx.x: 현재 블록의 인덱스 (0, 1, 2)
    # - thread_idx.x: 블록 내 스레드 인덱스 (0, 1, 2, 3)
    i = block_dim.x * block_idx.x + thread_idx.x

    # 경계 검사: 데이터 크기보다 큰 인덱스는 처리하지 않습니다
    # 중요한 이유:
    # 1. 총 스레드 수 (3블록 × 4스레드 = 12개) > 데이터 크기 (9개)
    # 2. 블록 2의 스레드 1,2,3은 인덱스 9,10,11을 가지지만 데이터는 0~8만 존재
    # 3. 이런 스레드들이 메모리에 접근하면 오류 발생
    if i < size:
        # 각 스레드가 담당하는 하나의 요소에 10을 더합니다
        # 메모리 접근 패턴: 연속적(coalesced) 접근으로 효율적
        # 블록 내 스레드들이 연속된 메모리 위치에 접근하여 메모리 대역폭 최적화
        output[i] = a[i] + 10.0


# ANCHOR_END: add_10_blocks


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # 이 컨텍스트는 GPU 메모리 할당, 데이터 복사, 커널 실행 등을 담당합니다
    with DeviceContext() as ctx:
        # 1D 배열을 위한 GPU 메모리 버퍼들을 생성합니다
        # SIZE = 9개의 요소를 가진 1차원 배열
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 출력 결과 버퍼
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 입력 배열 버퍼

        # 입력 배열을 초기화하기 위해 GPU 메모리를 호스트 메모리로 매핑합니다
        with a.map_to_host() as a_host:
            # 입력 배열을 0부터 8까지의 값으로 초기화합니다
            # a = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            for i in range(SIZE):
                a_host[i] = i

        # GPU 커널 함수를 실행합니다
        # 🔑 핵심: 다중 블록 구성으로 큰 데이터를 처리합니다
        #
        # 블록 구성 분석:
        # - 총 데이터: 9개 요소 [0,1,2,3,4,5,6,7,8]
        # - 블록 수: 3개 (BLOCKS_PER_GRID = (3,1))
        # - 블록당 스레드: 4개 (THREADS_PER_BLOCK = (4,1))
        # - 총 스레드: 3 × 4 = 12개 (데이터보다 많음!)
        #
        # 작업 분배:
        # 블록 0: 스레드 0,1,2,3 → 데이터 인덱스 0,1,2,3 처리
        # 블록 1: 스레드 0,1,2,3 → 데이터 인덱스 4,5,6,7 처리
        # 블록 2: 스레드 0,1,2,3 → 데이터 인덱스 8,9,10,11 시도
        #         (하지만 9,10,11은 범위 초과로 경계 검사에서 제외됨)
        ctx.enqueue_function[add_10_blocks](
            out.unsafe_ptr(),  # 출력 버퍼의 메모리 포인터
            a.unsafe_ptr(),  # 입력 배열의 메모리 포인터
            SIZE,  # 배열 크기 (9)
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (3개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (4개 스레드)
        )

        # 기대값을 저장할 호스트 메모리 버퍼를 생성합니다
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        ctx.synchronize()

        # 기대값을 계산합니다 (각 요소에 10을 더한 값)
        # expected = [10, 11, 12, 13, 14, 15, 16, 17, 18]
        for i in range(SIZE):
            expected[i] = i + 10

        # GPU에서 계산된 결과를 호스트 메모리로 매핑하여 확인합니다
        with out.map_to_host() as out_host:
            print(
                "out:", out_host
            )  # GPU 결과: [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
            print(
                "expected:", expected
            )  # 기대값: [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]

            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
