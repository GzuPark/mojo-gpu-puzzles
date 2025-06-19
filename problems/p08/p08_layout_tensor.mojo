from memory import UnsafePointer  # 메모리 포인터를 다루기 위한 클래스
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

# ANCHOR: add_10_shared_layout_tensor
# 프로그램에서 사용할 상수들을 정의합니다 (컴파일 타임에 결정됨)
alias TPB = 4  # Threads Per Block - 블록당 스레드 수 (4개)
alias SIZE = 8  # 1D 배열의 크기 (8개 요소)
alias BLOCKS_PER_GRID = (2, 1)  # 그리드당 블록 수 (2개 블록 사용)
alias THREADS_PER_BLOCK = (TPB, 1)  # 블록당 스레드 수 (4개 스레드)
alias dtype = DType.float32  # 데이터 타입 (32비트 부동소수점)
alias layout = Layout.row_major(SIZE)  # 행 우선(row-major) 레이아웃 정의


# GPU 커널 함수: LayoutTensor를 사용한 공유 메모리 처리
# 핵심 차이점: Raw Memory Approach vs LayoutTensor Approach
#
# Raw Memory Approach (p08.mojo):
# - stack_allocation으로 수동 메모리 할당
# - AddressSpace.SHARED 명시적 지정
# - 인덱스 계산과 메모리 관리를 직접 처리
# - 낮은 수준의 메모리 제어, 더 많은 코드 필요
#
# LayoutTensor Approach (현재 파일):
# - LayoutTensorBuild를 통한 선언적 메모리 할당
# - .shared() 메서드로 간편한 공유 메모리 지정
# - 자동화된 레이아웃 관리와 타입 안전성
# - 고수준 추상화, 더 간결하고 안전한 코드
fn add_10_shared_layout_tensor[
    layout: Layout  # 컴파일 타임에 결정되는 텐서 레이아웃 매개변수
](
    output: LayoutTensor[mut=True, dtype, layout],  # 출력 LayoutTensor (가변)
    a: LayoutTensor[mut=True, dtype, layout],  # 입력 LayoutTensor (가변)
    size: Int,  # 1D 배열의 크기 (8개 요소)
):
    # LayoutTensorBuild를 사용한 공유 메모리 할당
    # LayoutTensorBuild는 텐서 생성을 위한 플루언트 인터페이스(Fluent Interface)를 제공합니다
    #
    # 빌더 패턴 체이닝:
    # 1. tb[dtype](): 데이터 타입 지정 (float32)
    # 2. .row_major[TPB](): 행 우선 레이아웃, 크기 TPB(4) 지정
    # 3. .shared(): 공유 메모리 주소 공간 지정 (AddressSpace.SHARED)
    # 4. .alloc(): 실제 메모리 할당 수행
    #
    # Raw Memory와 비교:
    # Raw: stack_allocation[TPB, Scalar[dtype], address_space=AddressSpace.SHARED]()
    # LayoutTensor: tb[dtype]().row_major[TPB]().shared().alloc()
    # → 더 읽기 쉽고 의도가 명확함
    shared = tb[dtype]().row_major[TPB]().shared().alloc()

    # 글로벌 스레드 인덱스 계산 (전체 데이터에서의 위치)
    # LayoutTensor를 사용해도 기본적인 GPU 프로그래밍 패턴은 동일합니다
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # 로컬 스레드 인덱스 (블록 내에서의 위치)
    # 공유 메모리 접근 시 사용하는 인덱스
    local_i = thread_idx.x

    # 글로벌 메모리에서 공유 메모리로 데이터 로드
    # LayoutTensor의 장점: 자연스러운 인덱싱
    # Raw Memory: shared[local_i] = a[global_i] (포인터 접근)
    # LayoutTensor: shared[local_i] = a[global_i] (텐서 인덱싱)
    # → 동일한 문법이지만 LayoutTensor는 경계 검사와 타입 안전성 제공
    if global_i < size:
        shared[local_i] = a[global_i]

    # 동기화 배리어(Synchronization Barrier)
    # LayoutTensor를 사용해도 동기화는 여전히 필요합니다
    # 공유 메모리 접근 패턴이 바뀌지 않기 때문입니다
    barrier()

    # 공유 메모리에서 데이터를 읽어 계산 수행
    # LayoutTensor의 이점:
    # 1. 타입 안전성: 컴파일 타임에 차원과 타입 검증
    # 2. 메모리 레이아웃 추상화: 레이아웃 변경 시 코드 수정 최소화
    # 3. 자동 최적화: 컴파일러가 메모리 접근 패턴 최적화 가능
    if global_i < size:
        output[global_i] = shared[local_i] + 10.0


# ANCHOR_END: add_10_shared_layout_tensor


def main():
    # DeviceContext를 사용하여 GPU와의 상호작용을 관리합니다
    # LayoutTensor를 사용해도 기본적인 GPU 메모리 관리는 동일합니다
    with DeviceContext() as ctx:
        # 1D 배열을 위한 GPU 메모리 버퍼들을 생성합니다
        # Raw Memory Approach와 동일한 방식으로 버퍼 생성
        out = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(0)  # 출력 결과 버퍼
        a = ctx.enqueue_create_buffer[dtype](SIZE).enqueue_fill(1)  # 입력 배열 버퍼

        # Raw Pointer → LayoutTensor 변환
        # Raw Memory Approach에서는 UnsafePointer를 직접 사용했지만,
        # LayoutTensor Approach에서는 포인터를 LayoutTensor로 래핑합니다
        #
        # 장점:
        # 1. 타입 안전성: 컴파일 타임에 레이아웃과 차원 검증
        # 2. 자연스러운 인덱싱: tensor[i] 형태로 직관적 접근
        # 3. 메모리 레이아웃 추상화: 레이아웃 변경 시 코드 수정 최소화
        # 4. 자동 최적화: 컴파일러가 메모리 접근 패턴 최적화
        out_tensor = LayoutTensor[dtype, layout](
            out.unsafe_ptr()
        )  # 출력 버퍼를 LayoutTensor로 래핑
        a_tensor = LayoutTensor[dtype, layout](
            a.unsafe_ptr()
        )  # 입력 버퍼를 LayoutTensor로 래핑

        # GPU 커널 함수를 실행합니다
        # LayoutTensor를 매개변수로 전달
        #
        # Raw Memory vs LayoutTensor 비교:
        # Raw Memory: 함수 매개변수가 UnsafePointer[Scalar[dtype]]
        # LayoutTensor: 함수 매개변수가 LayoutTensor[mut=True, dtype, layout]
        # → LayoutTensor는 레이아웃 정보를 포함하여 더 안전하고 표현력이 풍부함
        #
        # 제네릭 매개변수 [layout] 전달:
        # - 컴파일 타임에 레이아웃 정보를 커널에 전달
        # - 런타임 오버헤드 없이 최적화된 코드 생성
        ctx.enqueue_function[add_10_shared_layout_tensor[layout]](
            out_tensor,  # LayoutTensor 출력 (UnsafePointer 대신)
            a_tensor,  # LayoutTensor 입력 (UnsafePointer 대신)
            SIZE,  # 배열 크기
            grid_dim=BLOCKS_PER_GRID,  # 그리드 차원 (2개 블록)
            block_dim=THREADS_PER_BLOCK,  # 블록 차원 (4개 스레드)
        )

        # 기대값을 저장할 호스트 메모리 버퍼를 생성합니다
        # 결과 검증 방식은 Raw Memory Approach와 동일합니다
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE).enqueue_fill(11)

        # 모든 GPU 작업이 완료될 때까지 기다립니다
        ctx.synchronize()

        # GPU에서 계산된 결과를 호스트 메모리로 매핑하여 확인합니다
        # LayoutTensor를 사용해도 결과 검증은 원본 버퍼를 통해 수행합니다
        with out.map_to_host() as out_host:
            print(
                "out:", out_host
            )  # GPU 결과: [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]
            print(
                "expected:", expected
            )  # 기대값: [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]

            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
