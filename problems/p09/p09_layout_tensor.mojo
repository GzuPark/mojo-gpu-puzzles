from gpu import (
    thread_idx,
    block_idx,
    block_dim,
    barrier,
)  # GPU 스레드/블록 인덱스, 차원 정보, 동기화 배리어
from gpu.host import DeviceContext  # GPU 디바이스와 상호작용하기 위한 컨텍스트 클래스
from layout import Layout, LayoutTensor  # 텐서 레이아웃과 LayoutTensor 클래스
from layout.tensor_builder import LayoutTensorBuild as tb  # 텐서 빌더 (별칭: tb)
from testing import assert_equal

# ANCHOR: pooling_layout_tensor
alias TPB = 8  # Threads Per Block - 블록당 스레드 수 (8개)
alias SIZE = 8  # 1D 배열의 크기 (8개 요소)
alias BLOCKS_PER_GRID = (1, 1)  # 그리드당 블록 수 (1개 블록만 사용)
alias THREADS_PER_BLOCK = (TPB, 1)  # 블록당 스레드 수 (8개 스레드)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)
alias layout = Layout.row_major(SIZE)  # 행 우선(row-major) 레이아웃 정의


# GPU 커널 함수: LayoutTensor를 사용한 슬라이딩 윈도우 풀링(Pooling) 연산
#
# 풀링(Pooling)의 핵심 개념:
# 풀링은 "최근 3개 위치의 누적 합(running sum)"을 계산하는 연산입니다.
# 각 position i에서 [i-2, i-1, i] 범위의 원소들을 합산합니다.
#
# 예시: 입력 배열 [0, 1, 2, 3, 4, 5, 6, 7]
# - output[0] = a[0] = 0 (범위: [0])
# - output[1] = a[0] + a[1] = 0 + 1 = 1 (범위: [0, 1])
# - output[2] = a[0] + a[1] + a[2] = 0 + 1 + 2 = 3 (범위: [0, 1, 2])
# - output[3] = a[1] + a[2] + a[3] = 1 + 2 + 3 = 6 (범위: [1, 2, 3])
# - output[4] = a[2] + a[3] + a[4] = 2 + 3 + 4 = 9 (범위: [2, 3, 4])
# - ... 이런 식으로 슬라이딩 윈도우가 이동하며 계산
#
# 핵심 차이점: Raw Memory Approach vs LayoutTensor Approach
#
# Raw Memory Approach (p09.mojo):
# - stack_allocation으로 수동 메모리 할당
# - AddressSpace.SHARED 명시적 지정
# - UnsafePointer[Scalar[dtype]] 매개변수 사용
# - 낮은 수준의 메모리 제어, 더 많은 코드 필요
# - 메모리 관리와 인덱싱을 직접 처리
#
# LayoutTensor Approach (현재 파일):
# - LayoutTensorBuild를 통한 선언적 메모리 할당
# - .shared() 메서드로 간편한 공유 메모리 지정
# - LayoutTensor[mut=True, dtype, layout] 매개변수 사용
# - 고수준 추상화, 더 간결하고 안전한 코드
# - 자동화된 레이아웃 관리와 타입 안전성
#
# 효율적인 구현 전략 (두 접근법 공통):
# 1. 공유 메모리 사용: 각 스레드가 데이터를 공유 메모리로 로드
# 2. 1 Global Read per Thread: 각 스레드가 글로벌 메모리에서 1번만 읽기
# 3. 1 Global Write per Thread: 각 스레드가 글로벌 메모리에 1번만 쓰기
# 4. 공유 메모리에서 인접 데이터에 접근하여 효율적인 풀링 연산 수행
fn pooling[
    layout: Layout  # 컴파일 타임에 결정되는 텐서 레이아웃 매개변수
](
    output: LayoutTensor[
        mut=True, dtype, layout
    ],  # 출력 LayoutTensor (가변, 풀링 결과 저장)
    a: LayoutTensor[mut=True, dtype, layout],  # 입력 LayoutTensor (가변, 원본 데이터)
    size: Int,  # 배열 크기 (8개 요소)
):
    # 공유 메모리(Shared Memory) 할당 - LayoutTensor 방식
    #
    # LayoutTensor의 공유 메모리 할당 장점:
    # 1. 선언적 문법: tb[dtype]().row_major[TPB]().shared().alloc()
    # 2. 빌더 패턴: 메서드 체이닝으로 가독성 향상
    # 3. 타입 안전성: 컴파일 타임에 레이아웃과 차원 검증
    # 4. 자동 최적화: 컴파일러가 메모리 접근 패턴 최적화
    #
    # Raw Memory vs LayoutTensor 비교:
    # Raw Memory: stack_allocation[TPB, Scalar[dtype], address_space=AddressSpace.SHARED]()
    # LayoutTensor: tb[dtype]().row_major[TPB]().shared().alloc()
    # → LayoutTensor는 더 읽기 쉽고 의도가 명확함
    #
    # 빌더 패턴 체이닝:
    # 1. tb[dtype](): 데이터 타입 지정 (float32)
    # 2. .row_major[TPB](): 행 우선 레이아웃, 크기 TPB(8) 지정
    # 3. .shared(): 공유 메모리 주소 공간 지정 (AddressSpace.SHARED와 동일)
    # 4. .alloc(): 실제 메모리 할당 수행
    shared = tb[dtype]().row_major[TPB]().shared().alloc()

    # 스레드 인덱스 계산 (Raw Memory Approach와 동일)
    # LayoutTensor를 사용해도 기본적인 GPU 프로그래밍 패턴은 동일합니다
    # 글로벌 스레드 인덱스: 전체 데이터에서의 위치
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # 로컬 스레드 인덱스: 블록 내에서의 위치 (공유 메모리 인덱스로 사용)
    local_i = thread_idx.x

    # FIX ME IN (roughly 10 lines)

    # 1단계: 글로벌 메모리에서 공유 메모리로 데이터 로드 (Cooperative Loading)
    # 각 스레드가 자신의 담당 데이터를 공유 메모리로 복사합니다.
    # 이렇게 하면 모든 스레드가 블록 내의 모든 데이터에 빠르게 접근할 수 있습니다.
    #
    # LayoutTensor의 인덱싱 장점:
    # Raw Memory: shared[local_i] = a[global_i] (포인터 접근)
    # LayoutTensor: shared[local_i] = a[global_i] (텐서 인덱싱)
    # → 동일한 문법이지만 LayoutTensor는 경계 검사와 타입 안전성 제공
    #
    # 경계 검사: global_i < size로 유효한 데이터 범위인지 확인
    # 만약 스레드 수가 데이터 수보다 많다면 일부 스레드는 유효하지 않은 데이터를 처리할 수 있음
    if global_i < size:
        shared[local_i] = a[global_i]

    # 동기화 배리어(Synchronization Barrier)
    # 모든 스레드가 데이터 로딩을 완료할 때까지 기다립니다.
    # LayoutTensor를 사용해도 동기화는 여전히 필요합니다
    # 공유 메모리 접근 패턴이 바뀌지 않기 때문입니다
    #
    # 배리어가 필요한 이유:
    # 1. 데이터 일관성: 모든 스레드가 공유 메모리에 데이터를 완전히 로드한 후에 읽기 시작
    # 2. 경쟁 조건 방지: 일부 스레드가 아직 데이터를 쓰고 있는 상태에서 다른 스레드가 읽는 것을 방지
    # 3. 메모리 순서 보장: GPU의 메모리 모델에서 쓰기와 읽기 순서를 보장
    barrier()

    # 2단계: 슬라이딩 윈도우 풀링 연산 수행
    # 각 스레드가 자신의 위치에서 최대 3개 원소의 합을 계산합니다.
    #
    # LayoutTensor의 연산 장점:
    # 1. 타입 안전성: 컴파일 타임에 차원과 타입 검증
    # 2. 메모리 레이아웃 추상화: 레이아웃 변경 시 코드 수정 최소화
    # 3. 자동 최적화: 컴파일러가 메모리 접근 패턴 최적화 가능
    # 4. 표현력: 텐서 연산의 의도가 더 명확하게 드러남
    #
    # 경우별 처리 (Edge Cases 고려) - Raw Memory와 동일한 로직:
    # 1. global_i == 0: 첫 번째 원소 (윈도우 크기 1)
    # 2. global_i == 1: 두 번째 원소 (윈도우 크기 2)
    # 3. global_i >= 2: 일반적인 경우 (윈도우 크기 3)

    if global_i == 0:
        # 첫 번째 위치: 윈도우 [0]
        # 첫 번째 원소는 자기 자신만 포함 (이전 원소가 없음)
        # LayoutTensor: output[0] = shared[0] (자연스러운 텐서 인덱싱)
        output[0] = shared[0]
    elif global_i == 1:
        # 두 번째 위치: 윈도우 [0, 1]
        # 두 번째 원소는 첫 번째와 두 번째 원소의 합
        # LayoutTensor: 텐서 연산이 더 직관적으로 표현됨
        output[1] = shared[0] + shared[1]
    elif 1 < global_i < size:
        # 일반적인 경우: 윈도우 [i-2, i-1, i]
        # 세 번째 원소부터는 이전 2개 + 현재 원소의 합을 계산
        #
        # 인덱스 매핑 (Raw Memory와 동일):
        # - shared[local_i - 2]: 현재 위치에서 2칸 앞의 원소
        # - shared[local_i - 1]: 현재 위치에서 1칸 앞의 원소
        # - shared[local_i]: 현재 위치의 원소
        #
        # 예시: global_i = 3일 때
        # - shared[1] + shared[2] + shared[3] = a[1] + a[2] + a[3]
        #
        # LayoutTensor의 연산 표현력:
        # Raw Memory와 동일한 계산이지만 LayoutTensor는 텐서 연산의 의미가 더 명확함
        output[global_i] = (
            shared[local_i - 2] + shared[local_i - 1] + shared[local_i]
        )


# ANCHOR_END: pooling_layout_tensor


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # LayoutTensor를 사용해도 기본적인 GPU 메모리 관리는 동일합니다
    with DeviceContext() as ctx:
        # GPU 메모리 버퍼 생성 및 초기화
        # Raw Memory Approach와 동일한 방식으로 버퍼 생성
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
            0
        )  # 출력 결과 버퍼 (0으로 초기화)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
            0
        )  # 입력 배열 버퍼 (0으로 초기화)

        # 입력 데이터 초기화: [0, 1, 2, 3, 4, 5, 6, 7]
        # 호스트에서 데이터를 생성하고 GPU 메모리로 전송합니다
        # 데이터 초기화 방식은 Raw Memory Approach와 동일합니다
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i  # 각 위치에 인덱스 값 저장

        # 핵심 차이점: Raw Pointer → LayoutTensor 변환
        # Raw Memory Approach에서는 UnsafePointer를 직접 사용했지만,
        # LayoutTensor Approach에서는 포인터를 LayoutTensor로 래핑합니다
        #
        # LayoutTensor 래핑의 장점:
        # 1. 타입 안전성: 컴파일 타임에 레이아웃과 차원 검증
        # 2. 자연스러운 인덱싱: tensor[i] 형태로 직관적 접근
        # 3. 메모리 레이아웃 추상화: 레이아웃 변경 시 코드 수정 최소화
        # 4. 자동 최적화: 컴파일러가 메모리 접근 패턴 최적화
        # 5. 표현력: 텐서 연산의 의도가 더 명확하게 드러남
        out_tensor = LayoutTensor[dtype, layout](
            out.unsafe_ptr()
        )  # 출력 버퍼를 LayoutTensor로 래핑
        a_tensor = LayoutTensor[dtype, layout](
            a.unsafe_ptr()
        )  # 입력 버퍼를 LayoutTensor로 래핑

        # GPU 커널 함수 실행 - LayoutTensor 매개변수 전달
        # 풀링 연산을 수행하는 커널을 GPU에서 실행합니다
        #
        # Raw Memory vs LayoutTensor 비교:
        # Raw Memory: 함수 매개변수가 UnsafePointer[Scalar[dtype]]
        # LayoutTensor: 함수 매개변수가 LayoutTensor[mut=True, dtype, layout]
        # → LayoutTensor는 레이아웃 정보를 포함하여 더 안전하고 표현력이 풍부함
        #
        # 제네릭 매개변수 [layout] 전달:
        # - 컴파일 타임에 레이아웃 정보를 커널에 전달
        # - 런타임 오버헤드 없이 최적화된 코드 생성
        # - 레이아웃 변경 시 자동으로 최적화된 코드 재생성
        #
        # 실행 구성:
        # - grid_dim=(1, 1): 1개의 블록만 사용
        # - block_dim=(8, 1): 블록당 8개의 스레드 사용
        # - 각 스레드가 1개의 position을 담당하여 병렬 처리
        ctx.enqueue_function[pooling[layout]](
            out_tensor,  # LayoutTensor 출력 (UnsafePointer 대신)
            a_tensor,  # LayoutTensor 입력 (UnsafePointer 대신)
            SIZE,  # 배열 크기
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (1개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (8개 스레드)
        )

        # 기대값 계산을 위한 호스트 메모리 버퍼 생성
        # 결과 검증 방식은 Raw Memory Approach와 동일합니다
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(0)

        # GPU 작업 완료 대기
        ctx.synchronize()

        # CPU에서 기대값 계산 (검증용)
        # GPU 결과와 비교하기 위해 CPU에서 동일한 풀링 연산을 수행합니다
        # 검증 로직은 Raw Memory Approach와 완전히 동일합니다
        with a.map_to_host() as a_host:
            ptr = a_host.unsafe_ptr()
            for i in range(SIZE):
                s = Scalar[dtype](0)  # 누적 합 초기화

                # 슬라이딩 윈도우 범위 계산: [max(i-2, 0), i]
                # - max(i-2, 0): 음수 인덱스 방지 (배열 시작 경계 처리)
                # - i+1: Python의 range() 함수는 끝 값을 포함하지 않으므로 +1
                #
                # 예시:
                # - i=0: range(0, 1) → [0]
                # - i=1: range(0, 2) → [0, 1]
                # - i=2: range(0, 3) → [0, 1, 2]
                # - i=3: range(1, 4) → [1, 2, 3]
                for j in range(max(i - 2, 0), i + 1):
                    s += ptr[j]
                expected[i] = s  # 계산된 기대값 저장

        with out.map_to_host() as out_host:
            print("out:", out_host)  # GPU 풀링 결과
            print("expected:", expected)  # CPU 계산 기대값
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
