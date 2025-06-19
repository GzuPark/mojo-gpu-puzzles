from testing import assert_equal
from gpu.host import DeviceContext

# ANCHOR: dot_product_layout_tensor
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb


alias TPB = 8  # Threads Per Block - 블록당 스레드 수 (8개)
alias SIZE = 8  # 1D 배열의 크기 (8개 요소)
alias BLOCKS_PER_GRID = (1, 1)  # 그리드당 블록 수 (1개 블록만 사용)
alias THREADS_PER_BLOCK = (SIZE, 1)  # 블록당 스레드 수 (SIZE=8개 스레드)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)
alias layout = Layout.row_major(SIZE)  # 입력 벡터용 행 우선(row-major) 레이아웃 정의
alias out_layout = Layout.row_major(1)  # 출력 스칼라용 행 우선 레이아웃 정의 (크기 1)


# GPU 커널 함수: LayoutTensor를 사용한 내적(Dot Product) 연산
#
# 내적(Dot Product)의 핵심 개념:
# 내적은 두 벡터의 대응하는 원소들을 곱한 후 모든 결과를 더하는 연산입니다.
# 수학적 표현: a · b = a₀×b₀ + a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ
#
# 예시: 벡터 a = [0, 1, 2, 3, 4, 5, 6, 7], 벡터 b = [0, 1, 2, 3, 4, 5, 6, 7]
# 최종 결과: 0×0 + 1×1 + 2×2 + 3×3 + 4×4 + 5×5 + 6×6 + 7×7 = 140
#
# 핵심 차이점: Raw Memory Approach vs LayoutTensor Approach
#
# Raw Memory Approach (p10.mojo):
# - stack_allocation으로 수동 메모리 할당
# - AddressSpace.SHARED 명시적 지정
# - UnsafePointer[Scalar[dtype]] 매개변수 사용
# - 단일 레이아웃 처리 (입력과 출력이 동일한 포인터 타입)
# - 낮은 수준의 메모리 제어
#
# LayoutTensor Approach (현재 파일):
# - LayoutTensorBuild를 통한 선언적 메모리 할당
# - .shared() 메서드로 간편한 공유 메모리 지정
# - LayoutTensor[mut=True, dtype, layout] 매개변수 사용
# - 다중 레이아웃 지원 (입력과 출력에 서로 다른 레이아웃 적용 가능)
# - 고수준 추상화, 더 간결하고 안전한 코드
# - 레이아웃 인식 연산 (Layout-aware Operations)
#
# 다중 레이아웃 매개변수 (Multiple Layout Parameters)
# 이전 LayoutTensor 예제들과 달리 입력과 출력에 서로 다른 레이아웃을 사용합니다:
# - in_layout: 입력 벡터용 레이아웃 (크기 SIZE=8)
# - out_layout: 출력 스칼라용 레이아웃 (크기 1)
# 이를 통해 다양한 차원과 형태의 텐서를 효율적으로 처리할 수 있습니다.
fn dot_product[
    in_layout: Layout,  # 입력 벡터용 레이아웃 매개변수 (컴파일 타임 결정)
    out_layout: Layout,  # 출력 스칼라용 레이아웃 매개변수 (컴파일 타임 결정)
](
    output: LayoutTensor[
        mut=True, dtype, out_layout
    ],  # 출력 LayoutTensor (가변, 내적 결과 저장, 크기 1)
    a: LayoutTensor[
        mut=True, dtype, in_layout
    ],  # 첫 번째 입력 LayoutTensor (가변, 크기 SIZE)
    b: LayoutTensor[
        mut=True, dtype, in_layout
    ],  # 두 번째 입력 LayoutTensor (가변, 크기 SIZE)
    size: Int,  # 벡터 크기 (8개 요소)
):
    # FILL ME IN (roughly 13 lines)

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

    # 1단계: 원소별 곱셈 및 공유 메모리 로드 (Element-wise Multiplication)
    # 각 스레드가 대응하는 원소들을 곱하고 결과를 공유 메모리에 저장합니다.
    #
    # LayoutTensor의 자연스러운 인덱싱 장점:
    # Raw Memory: shared[local_i] = a[global_i] * b[global_i] (포인터 접근)
    # LayoutTensor: shared[local_i] = a[global_i] * b[global_i] (텐서 인덱싱)
    # → 동일한 문법이지만 LayoutTensor는 경계 검사와 타입 안전성 제공
    #
    # 병렬 처리 패턴:
    # - Thread 0: a[0] × b[0] → shared[0]
    # - Thread 1: a[1] × b[1] → shared[1]
    # - Thread 2: a[2] × b[2] → shared[2]
    # - ... 모든 스레드가 동시에 실행
    #
    # 경계 검사: global_i < size로 유효한 데이터 범위인지 확인
    if global_i < size:
        shared[local_i] = a[global_i] * b[global_i]

    # 동기화 배리어(Synchronization Barrier)
    # 모든 스레드가 원소별 곱셈을 완료할 때까지 기다립니다.
    # LayoutTensor를 사용해도 동기화는 여전히 필요합니다
    # 공유 메모리 접근 패턴이 바뀌지 않기 때문입니다
    #
    # 배리어가 필요한 이유:
    # 1. 데이터 일관성: 모든 곱셈 결과가 공유 메모리에 완전히 저장된 후 리덕션 시작
    # 2. 경쟁 조건 방지: 일부 스레드가 아직 쓰기 중인 데이터를 다른 스레드가 읽는 것 방지
    # 3. 단계별 동기화: 리덕션의 각 단계마다 모든 스레드가 동기화되어야 함
    barrier()

    # 2단계: 트리 기반 리덕션(Tree-based Reduction) 수행
    # 이진 트리 구조로 병렬 합산을 수행하여 최종 내적 결과를 계산합니다.
    #
    # LayoutTensor의 리덕션 연산 장점:
    # 1. 타입 안전성: 컴파일 타임에 차원과 타입 검증
    # 2. 메모리 레이아웃 추상화: 레이아웃 변경 시 코드 수정 최소화
    # 3. 자동 최적화: 컴파일러가 메모리 접근 패턴 최적화 가능
    # 4. 표현력: 텐서 연산의 의도가 더 명확하게 드러남
    #
    # 리덕션 알고리즘 (Raw Memory와 동일한 로직):
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
        if local_i < stride:
            # 쌍별 합산(Pairwise Addition) 수행
            # 현재 위치의 값과 stride만큼 떨어진 위치의 값을 합산
            #
            # LayoutTensor의 연산 표현력:
            # Raw Memory와 동일한 계산이지만 LayoutTensor는 텐서 연산의 의미가 더 명확함
            # 메모리 접근 패턴이 레이아웃에 의해 추상화되어 안전성 향상
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
    # LayoutTensor의 출력 장점:
    # Raw Memory: output[0] = shared[0] (포인터 접근)
    # LayoutTensor: output[0] = shared[0] (텐서 인덱싱)
    # → 동일한 문법이지만 LayoutTensor는 out_layout에 따른 타입 안전성 제공
    #
    # 왜 Thread 0만 쓰기를 하는가?
    # 1. 중복 쓰기 방지: 여러 스레드가 동시에 쓰면 비효율적
    # 2. 메모리 일관성: 단일 스레드 쓰기로 데이터 무결성 보장
    # 3. 효율성: 1 Global Write per Block 달성
    if local_i == 0:
        output[0] = shared[0]  # 최종 내적 결과를 출력 텐서에 저장


# ANCHOR_END: dot_product_layout_tensor


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # LayoutTensor를 사용해도 기본적인 GPU 메모리 관리는 동일합니다
    with DeviceContext() as ctx:
        # GPU 메모리 버퍼 생성 및 초기화
        # Raw Memory Approach와 동일한 방식으로 버퍼 생성
        out = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(
            0
        )  # 출력 결과 버퍼 (크기 1, 내적 결과)
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
            0
        )  # 첫 번째 벡터 버퍼 (크기 SIZE)
        b = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(
            0
        )  # 두 번째 벡터 버퍼 (크기 SIZE)

        # 입력 데이터 초기화: a = [0, 1, 2, 3, 4, 5, 6, 7], b = [0, 1, 2, 3, 4, 5, 6, 7]
        # 데이터 초기화 방식은 Raw Memory Approach와 동일합니다
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i  # 첫 번째 벡터: [0, 1, 2, 3, 4, 5, 6, 7]
                b_host[i] = i  # 두 번째 벡터: [0, 1, 2, 3, 4, 5, 6, 7]

        # 핵심 차이점: Raw Pointer → LayoutTensor 변환 (다중 레이아웃)
        # Raw Memory Approach에서는 UnsafePointer를 직접 사용했지만,
        # LayoutTensor Approach에서는 포인터를 LayoutTensor로 래핑합니다
        #
        # 새로운 개념: 서로 다른 레이아웃 적용
        # - 입력 벡터: layout (크기 SIZE=8)
        # - 출력 스칼라: out_layout (크기 1)
        # 이를 통해 입력과 출력의 차원이 다른 연산을 타입 안전하게 처리할 수 있습니다
        #
        # LayoutTensor 래핑의 장점:
        # 1. 타입 안전성: 컴파일 타임에 레이아웃과 차원 검증
        # 2. 자연스러운 인덱싱: tensor[i] 형태로 직관적 접근
        # 3. 메모리 레이아웃 추상화: 레이아웃 변경 시 코드 수정 최소화
        # 4. 자동 최적화: 컴파일러가 메모리 접근 패턴 최적화
        # 5. 다중 레이아웃 지원: 입력과 출력에 서로 다른 레이아웃 적용 가능
        out_tensor = LayoutTensor[dtype, out_layout](
            out.unsafe_ptr()
        )  # 출력 버퍼를 LayoutTensor로 래핑 (out_layout 사용)
        a_tensor = LayoutTensor[dtype, layout](
            a.unsafe_ptr()
        )  # 첫 번째 입력 버퍼를 LayoutTensor로 래핑 (layout 사용)
        b_tensor = LayoutTensor[dtype, layout](
            b.unsafe_ptr()
        )  # 두 번째 입력 버퍼를 LayoutTensor로 래핑 (layout 사용)

        # GPU 커널 함수 실행 - 다중 레이아웃 매개변수 전달
        # 내적 연산을 수행하는 커널을 GPU에서 실행합니다
        #
        # Raw Memory vs LayoutTensor 비교:
        # Raw Memory: 함수 매개변수가 UnsafePointer[Scalar[dtype]]
        # LayoutTensor: 함수 매개변수가 LayoutTensor[mut=True, dtype, layout]
        # → LayoutTensor는 레이아웃 정보를 포함하여 더 안전하고 표현력이 풍부함
        #
        # 다중 제네릭 매개변수 전달
        # - [layout, out_layout]: 입력과 출력에 서로 다른 레이아웃 정보 전달
        # - 컴파일 타임에 두 레이아웃 정보를 모두 커널에 전달
        # - 런타임 오버헤드 없이 최적화된 코드 생성
        # - 레이아웃 변경 시 자동으로 최적화된 코드 재생성
        #
        # 실행 구성 (Raw Memory와 동일):
        # - grid_dim=(1, 1): 1개의 블록만 사용 (리덕션 특성)
        # - block_dim=(SIZE, 1): 블록당 SIZE(8)개의 스레드 사용
        # - 각 스레드가 1개의 원소 쌍을 담당하여 병렬 곱셈 수행
        ctx.enqueue_function[dot_product[layout, out_layout]](
            out_tensor,  # LayoutTensor 출력 (out_layout 적용)
            a_tensor,  # LayoutTensor 첫 번째 입력 (layout 적용)
            b_tensor,  # LayoutTensor 두 번째 입력 (layout 적용)
            SIZE,  # 벡터 크기
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (1개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (8개 스레드)
        )

        # 기대값 계산을 위한 호스트 메모리 버퍼 생성
        # 결과 검증 방식은 Raw Memory Approach와 동일합니다
        expected = ctx.enqueue_create_host_buffer[dtype](1).enqueue_fill(0)

        # GPU 작업 완료 대기
        ctx.synchronize()

        # CPU에서 기대값 계산 (검증용)
        # GPU 결과와 비교하기 위해 CPU에서 동일한 내적 연산을 수행합니다
        # 검증 로직은 Raw Memory Approach와 완전히 동일합니다
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            # 순차적 내적 계산: 각 원소 쌍을 곱하고 누적 합산
            for i in range(SIZE):
                expected[0] += a_host[i] * b_host[i]

        with out.map_to_host() as out_host:
            print("out:", out_host)  # GPU 내적 결과 (LayoutTensor 트리 리덕션)
            print("expected:", expected)  # CPU 내적 결과 (순차적 계산)
            assert_equal(out_host[0], expected[0])
