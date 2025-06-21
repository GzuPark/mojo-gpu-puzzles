from sys import sizeof, argv
from testing import assert_equal
from gpu.host import DeviceContext

# ANCHOR: naive_matmul
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb

# Matrix Multiplication (행렬 곱셈) - GPU 최적화의 핵심
#
# 두 행렬 A(m×k)와 B(k×n)을 곱하여 결과 행렬 C(m×n)을 생성하는 연산입니다.
#
# 수학적 정의: C[i,j] = Σ(l=0 to k-1) A[i,l] × B[l,j]
#
# 예시: 2×2 행렬 곱셈
# A = [0 1]    B = [0 2]    C = A×B = [4  6 ]
#     [2 3]        [4 6]              [12 22]
#
# 계산 과정:
# C[0,0] = A[0,0]×B[0,0] + A[0,1]×B[1,0] = 0×0 + 1×4 = 4
# C[0,1] = A[0,0]×B[0,1] + A[0,1]×B[1,1] = 0×2 + 1×6 = 6
# C[1,0] = A[1,0]×B[0,0] + A[1,1]×B[1,0] = 2×0 + 3×4 = 12
# C[1,1] = A[1,0]×B[0,1] + A[1,1]×B[1,1] = 2×2 + 3×6 = 22
#
# GPU에서 Matrix Multiplication의 중요성:
# 1. 머신러닝/딥러닝의 핵심 연산 (신경망의 기본 구성 요소)
# 2. 과학 계산의 기본 연산 (선형대수, 시뮬레이션)
# 3. GPU 최적화 기법의 집약체 (메모리 계층, 병렬성, 데이터 재사용)
# 4. 성능 분석의 표준 벤치마크 (FLOPS, 메모리 대역폭 측정)

# Roofline Model 이론 - GPU 성능 분석의 핵심 도구
#
# GPU 커널의 성능을 분석하고 최적화 방향을 제시하는 시각적 성능 모델입니다.
#
# 핵심 개념:
# 1. Arithmetic Intensity (연산 강도): I = FLOPs / Bytes
#    - 메모리에서 읽은 바이트당 수행하는 부동소수점 연산 수
#    - 단위: FLOP/B (Floating Point Operations per Byte)
#
# 2. Sustained Performance (지속 성능): P = FLOPs / Time
#    - 실제로 달성한 연산 처리량
#    - 단위: GFLOP/s (Giga Floating Point Operations per Second)
#
# 3. 두 가지 성능 한계 (Performance Ceilings):
#    a) Memory Roof (메모리 루프): P = B_peak × I
#       - 기울기가 있는 직선 (메모리 대역폭에 의한 제한)
#       - 메모리 바운드 영역에서 성능 상한선
#
#    b) Compute Roof (컴퓨트 루프): P = P_peak
#       - 수평선 (컴퓨팅 처리량에 의한 제한)
#       - 컴퓨트 바운드 영역에서 성능 상한선
#
# 4. Critical Intensity (임계 강도): I* = P_peak / B_peak
#    - 메모리 바운드에서 컴퓨트 바운드로 전환되는 지점
#    - I < I*: 메모리 바운드 (메모리 대역폭이 병목)
#    - I > I*: 컴퓨트 바운드 (컴퓨팅 처리량이 병목)
#
# NVIDIA A100 GPU 사양 (참고):
# - Peak FP32 Performance: P_peak = 19.5 TFLOP/s
# - Peak Memory Bandwidth: B_peak = 1,555 GB/s
# - Critical Intensity: I* = 19.5 / 1.555 ≈ 12.5 FLOP/B
#
# Roofline Model 활용법:
# 1. 커널의 (I, P) 좌표를 그래프에 표시
# 2. 어느 루프(메모리/컴퓨트)에 위치하는지 확인
# 3. 병목 지점에 따른 최적화 전략 수립:
#    - 메모리 바운드: 메모리 접근 최적화 (공유 메모리, 캐싱, 코얼레싱)
#    - 컴퓨트 바운드: 알고리즘 최적화 (병렬성 증대, 연산 효율성)
#
# Matrix Multiplication에서의 Roofline 적용:
# 세 가지 구현 방법이 Roofline 상에서 어떻게 이동하는지 분석하여
# 각 최적화 기법의 효과를 정량적으로 이해할 수 있습니다.

# Naive Implementation 구성 상수들
alias TPB = 3  # Threads Per Block (3×3 = 9개 스레드)
alias SIZE = 2  # 행렬 크기 (2×2 행렬)
alias BLOCKS_PER_GRID = (1, 1)  # 단일 블록 사용
alias THREADS_PER_BLOCK = (TPB, TPB)  # 2D 스레드 블록 (3×3)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)  # 2×2 행렬 레이아웃


# 방법 1: Naive Matrix Multiplication
#
# Naive 접근법의 특징:
# 1. 각 스레드가 출력 행렬의 하나의 원소를 계산
# 2. 글로벌 메모리에서 직접 데이터 읽기
# 3. 데이터 재사용 없음 (매번 글로벌 메모리 접근)
# 4. 구현이 간단하지만 메모리 효율성이 낮음
#
# Arithmetic Intensity 계산 (정확한 분석):
#
# 출력 원소 하나당:
# - FLOPs: SIZE개 곱셈 + (SIZE-1)개 덧셈 = 2×SIZE - 1 = 2×2 - 1 = 3 FLOPs
# - Memory Access:
#   * A 행렬에서 SIZE개 원소 읽기: SIZE × 4 bytes = 2 × 4 = 8 bytes
#   * B 행렬에서 SIZE개 원소 읽기: SIZE × 4 bytes = 2 × 4 = 8 bytes
#   * 출력에 1개 원소 쓰기: 1 × 4 bytes = 4 bytes
#   * 총 메모리 접근: 8 + 8 + 4 = 20 bytes
#
# Arithmetic Intensity = 3 FLOPs / 20 bytes = 0.15 FLOP/B
#
# Roofline Model 관점에서의 분석:
# - I = 0.15 FLOP/B << I* = 12.5 FLOP/B (심각한 메모리 바운드)
# - 예상 성능: P ≈ B_peak × I = 1,555 × 0.15 ≈ 233 GFLOP/s
# - GPU 피크 성능 대비 활용률: 233/19,500 ≈ 1.2% (매우 낮음)
# - 병목: 글로벌 메모리 대역폭 (데이터 재사용 없음)
#
# 메모리 접근 패턴의 문제점:
# - 각 출력 원소마다 A의 행과 B의 열을 중복해서 읽음
# - 동일한 데이터를 여러 번 글로벌 메모리에서 가져옴
# - 메모리 대역폭 낭비가 심각함
fn naive_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    # 2D 스레드 인덱싱: 행렬의 (row, col) 위치 계산
    #
    # 스레드-행렬 원소 매핑:
    # Thread(0,0) → Matrix[0,0]    Thread(0,1) → Matrix[0,1]    Thread(0,2) → 범위 초과
    # Thread(1,0) → Matrix[1,0]    Thread(1,1) → Matrix[1,1]    Thread(1,2) → 범위 초과
    # Thread(2,0) → 범위 초과       Thread(2,1) → 범위 초과       Thread(2,2) → 범위 초과
    #
    # 3×3 스레드 블록으로 2×2 행렬 처리 (일부 스레드는 유휴 상태)
    row = block_dim.y * block_idx.y + thread_idx.y  # 행 인덱스
    col = block_dim.x * block_idx.x + thread_idx.x  # 열 인덱스

    # FILL ME IN (roughly 6 lines)

    # 경계 검사: 유효한 행렬 원소만 처리
    if row < size and col < size:
        # 누적 변수 초기화 (출력 텐서와 동일한 타입 사용)
        var acc: output.element_type = 0

        # 내적 계산: A의 row번째 행과 B의 col번째 열의 내적
        #
        # 메모리 접근 패턴 (Thread(0,0) 예시):
        # k=0: a[0,0] × b[0,0]  (글로벌 메모리에서 2번 읽기)
        # k=1: a[0,1] × b[1,0]  (글로벌 메모리에서 2번 읽기)
        # 총 4번의 글로벌 메모리 읽기 + 1번 쓰기 = 20 bytes 메모리 접근
        @parameter
        for k in range(size):
            acc += a[row, k] * b[k, col]

        # 결과를 출력 행렬에 저장 (1번의 글로벌 메모리 쓰기)
        output[row, col] = acc


# ANCHOR_END: naive_matmul


# ANCHOR: single_block_matmul
# 방법 2: Shared Memory Matrix Multiplication
#
# Shared Memory 접근법의 특징:
# 1. 입력 행렬을 공유 메모리로 미리 로드
# 2. 빠른 공유 메모리에서 데이터 읽기
# 3. 글로벌 메모리 접근 횟수 감소
# 4. 데이터 재사용을 통한 성능 향상
#
# Arithmetic Intensity 계산 (정확한 분석):
#
# 블록 전체 관점에서 분석 (4개 출력 원소, 4개 활성 스레드):
# - 총 FLOPs: 4개 원소 × 3 FLOPs = 12 FLOPs
# - 글로벌 메모리 접근:
#   * A 행렬 로딩: 4개 원소 × 4 bytes = 16 bytes
#   * B 행렬 로딩: 4개 원소 × 4 bytes = 16 bytes
#   * 출력 쓰기: 4개 원소 × 4 bytes = 16 bytes
#   * 총 글로벌 메모리 접근: 16 + 16 + 16 = 48 bytes
#
# Arithmetic Intensity = 12 FLOPs / 48 bytes = 0.25 FLOP/B
#
# Roofline Model 관점에서의 개선:
# - I = 0.25 FLOP/B (Naive 대비 67% 증가: 0.25/0.15 = 1.67)
# - 예상 성능: P ≈ B_peak × I = 1,555 × 0.25 ≈ 389 GFLOP/s
# - GPU 피크 성능 대비 활용률: 389/19,500 ≈ 2.0% (개선됨)
# - 여전히 메모리 바운드이지만 성능 향상 달성
#
# 성능 개선의 원리:
# 1. 데이터 재사용: 공유 메모리에 로드된 데이터를 여러 스레드가 활용
# 2. 메모리 계층 활용: 글로벌 메모리(느림) → 공유 메모리(빠름)
# 3. 협력적 로딩: 모든 스레드가 협력하여 메모리 대역폭 최대 활용
#
# 메모리 계층 성능 비교:
# - 글로벌 메모리: ~400-800 사이클 레이턴시, 1,555 GB/s 대역폭
# - 공유 메모리: ~1-2 사이클 레이턴시, 블록당 48KB 용량
# - 레지스터: ~1 사이클 레이턴시, 스레드당 255개 32-bit 레지스터
fn single_block_matmul[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    # 글로벌 및 로컬 인덱스 계산
    row = block_dim.y * block_idx.y + thread_idx.y  # 글로벌 행 인덱스
    col = block_dim.x * block_idx.x + thread_idx.x  # 글로벌 열 인덱스
    local_row = thread_idx.y  # 공유 메모리 내 행 인덱스
    local_col = thread_idx.x  # 공유 메모리 내 열 인덱스

    # FILL ME IN (roughly 12 lines)

    # 공유 메모리 할당: 각 행렬을 위한 TPB×TPB 크기
    #
    # 메모리 레이아웃:
    # a_shared[3×3]: A 행렬의 블록을 저장 (36 bytes)
    # b_shared[3×3]: B 행렬의 블록을 저장 (36 bytes)
    # 총 공유 메모리 사용량: 72 bytes (48KB 한도 내)
    #
    # 공유 메모리 vs 글로벌 메모리:
    # - 공유 메모리: 48KB (블록당), 1-2 사이클 레이턴시
    # - 글로벌 메모리: 수 GB, 400-800 사이클 레이턴시
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    # 1단계: 협력적 데이터 로딩 (Cooperative Loading)
    #
    # 각 스레드가 하나의 원소를 담당하여 글로벌 메모리에서 공유 메모리로 로드
    #
    # 로딩 패턴 (메모리 코얼레싱 최적화):
    # Thread(0,0): a[0,0] → a_shared[0,0], b[0,0] → b_shared[0,0]
    # Thread(0,1): a[0,1] → a_shared[0,1], b[0,1] → b_shared[0,1]
    # Thread(1,0): a[1,0] → a_shared[1,0], b[1,0] → b_shared[1,0]
    # Thread(1,1): a[1,1] → a_shared[1,1], b[1,1] → b_shared[1,1]
    #
    # 장점:
    # - 모든 스레드가 협력하여 메모리 대역폭 최대 활용
    # - 연속된 메모리 주소 접근으로 코얼레싱 효과
    # - 한 번의 로딩으로 모든 스레드가 데이터 공유
    if row < size and col < size:
        a_shared[local_row, local_col] = a[row, col]
        b_shared[local_row, local_col] = b[row, col]

    barrier()  # 모든 스레드의 로딩 완료 대기 (동기화 필수)

    # 2단계: 공유 메모리를 사용한 행렬 곱셈
    #
    # 성능 개선 효과 (정량적 분석):
    # - Naive: 각 원소마다 4번의 글로벌 메모리 접근 (20 bytes)
    # - Shared: 초기 로딩 후 빠른 공유 메모리에서 접근 (12 bytes 글로벌)
    # - 메모리 접근 시간: 400 사이클 → 1 사이클 (400배 빠름)
    # - 글로벌 메모리 트래픽: 67% 감소 (20 → 12 bytes per output element)
    if row < size and col < size:
        var acc: output.element_type = 0

        # 공유 메모리에서 내적 계산
        # a_shared[local_row, k]: A의 행 데이터 (공유 메모리, ~1 사이클)
        # b_shared[k, local_col]: B의 열 데이터 (공유 메모리, ~1 사이클)
        #
        # 데이터 재사용 효과:
        # - a_shared[0,0]은 출력의 [0,0], [0,1] 계산에 재사용
        # - b_shared[0,0]은 출력의 [0,0], [1,0] 계산에 재사용
        # - 메모리 대역폭 효율성 크게 향상
        @parameter
        for k in range(size):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        output[row, col] = acc


# ANCHOR_END: single_block_matmul

# ANCHOR: matmul_tiled
# 방법 3: Tiled Matrix Multiplication
#
# Tiled 접근법의 특징:
# 1. 큰 행렬을 작은 타일(tile)로 분할하여 처리
# 2. 다중 블록을 사용하여 대용량 행렬 처리
# 3. 타일별 순차 처리로 메모리 사용량 최적화
# 4. 실제 GPU 애플리케이션에서 사용하는 방법
#
# 🔍 GPU 블록 실행 모델과 타일 처리 방식 상세 설명
#
# ## 혼란의 원인: 두 가지 다른 개념
#
# ### 📊 개념 1: 블록 배치 (어떤 블록들이 있는가?)
# GPU 그리드: 3×3 = 9개 블록
# [Block(0,0)] [Block(0,1)] [Block(0,2)]
# [Block(1,0)] [Block(1,1)] [Block(1,2)]
# [Block(2,0)] [Block(2,1)] [Block(2,2)]
#
# ### 🔄 개념 2: 타일 처리 순서 (각 블록 내에서 어떤 순서로 계산하는가?)
# 각 블록 내에서:
# 1단계: 타일 0 처리 (k=0~2)
# 2단계: 타일 1 처리 (k=3~5)
# 3단계: 타일 2 처리 (k=6~7)
#
# ## 🚀 GPU 실행 모델: 실제로 어떻게 돌아가는가?
#
# ### 시간축 관점에서 보기
#
# ⏰ 시각 T0: GPU 커널 시작
# GPU 스케줄러가 9개 블록을 사용 가능한 SM(Streaming Multiprocessor)에 배치
# SM0: Block(0,0) 시작, SM1: Block(0,1) 시작, ... SM8: Block(2,2) 시작
# → 모든 블록이 동시에 시작됨!
#
# ⏰ 시각 T1: 모든 블록이 타일 0 처리
# 동시에 일어나는 일들:
# Block(0,0): A[0:3,0:3] × B[0:3,0:3] → C[0:3,0:3] 부분 계산
# Block(0,1): A[0:3,0:3] × B[0:3,3:6] → C[0:3,3:6] 부분 계산
# Block(1,1): A[3:6,0:3] × B[0:3,3:6] → C[3:6,3:6] 부분 계산
# ... 9개 블록이 각자 다른 데이터로 타일 0 계산을 동시에 수행!
#
# ⏰ 시각 T2: 모든 블록이 타일 1 처리
# Block(0,0): A[0:3,3:6] × B[3:6,0:3] → C[0:3,0:3] 부분 누적
# Block(0,1): A[0:3,3:6] × B[3:6,3:6] → C[0:3,3:6] 부분 누적
# ... 9개 블록이 각자 다른 데이터로 타일 1 계산을 동시에 수행!
#
# ⏰ 시각 T3: 모든 블록이 타일 2 처리
# Block(0,0): A[0:3,6:8] × B[6:8,0:3] → C[0:3,0:3] 최종 완성
# Block(0,1): A[0:3,6:8] × B[6:8,3:6] → C[0:3,3:6] 최종 완성
# ... 9개 블록이 각자의 최종 결과를 동시에 완성!
#
# ## 🔑 핵심 차이점 정리
#
# ❌ 잘못된 이해 (순차적 실행):
# 시간 T1: Block(0,0), Block(0,1), Block(0,2)만 실행 → C[0:3, :] 완성
# 시간 T2: Block(1,0), Block(1,1), Block(1,2)만 실행 → C[3:6, :] 완성
# 시간 T3: Block(2,0), Block(2,1), Block(2,2)만 실행 → C[6:8, :] 완성
# 문제점: GPU의 병렬성을 전혀 활용하지 못함
#
# ✅ 올바른 이해 (병렬 실행):
# 시간 T1-T3: 모든 9개 블록이 동시에 실행됨
# 각 블록은 독립적으로:
# - 자신의 출력 영역 담당
# - 동일한 3단계 타일 처리 수행
# - 서로 다른 입력 데이터 사용
#
# 💡 실제 GPU 하드웨어 관점:
# NVIDIA GPU 예시: SM(Streaming Multiprocessor) 108개, 동시 실행 가능 블록 수천 개
# 우리의 9개 블록: 9개 SM에서 동시 실행, 실행 시간 거의 동일하게 완료
# 따라서 모든 블록이 정말로 "동시에" 실행됩니다! 🚀
#
# Arithmetic Intensity 계산 (8×8 행렬, 정확한 분석):
#
# 전체 행렬 관점에서 분석:
# - 총 FLOPs: 8×8×(2×8-1) = 64 × 15 = 960 FLOPs
# - 글로벌 메모리 접근:
#   * A 행렬 읽기: 64개 원소 × 4 bytes = 256 bytes
#   * B 행렬 읽기: 64개 원소 × 4 bytes = 256 bytes
#   * 출력 쓰기: 64개 원소 × 4 bytes = 256 bytes
#   * 총 글로벌 메모리 접근: 256 + 256 + 256 = 768 bytes
#
# Arithmetic Intensity = 960 FLOPs / 768 bytes = 1.25 FLOP/B
#
# 타일별 세부 분석 (3×3 타일 기준):
# - 타일당 FLOPs: 3×3×(2×3-1) = 9 × 5 = 45 FLOPs
# - 타일당 글로벌 메모리 로딩: 2 × (3×3) × 4 bytes = 72 bytes
# - 타일 내 Arithmetic Intensity: 45 FLOPs / 72 bytes = 0.625 FLOP/B
#
# Roofline Model 관점에서의 최적화:
# - I = 1.25 FLOP/B (Naive 대비 8.3배 증가: 1.25/0.15 = 8.33)
# - 예상 성능: P ≈ B_peak × I = 1,555 × 1.25 ≈ 1,944 GFLOP/s
# - GPU 피크 성능 대비 활용률: 1,944/19,500 ≈ 10.0% (상당한 개선)
# - 여전히 메모리 바운드이지만 임계점에 근접
#
# 큰 행렬에서의 확장성:
# - 행렬 크기가 커질수록 Arithmetic Intensity 증가
# - 1024×1024 행렬: I ≈ 341 FLOP/B (컴퓨트 바운드 달성)
# - 실제 딥러닝 워크로드에서 높은 성능 달성 가능
#
# Tiling의 핵심 아이디어:
# - 큰 문제를 작은 문제들로 분해 (Divide and Conquer)
# - 각 타일은 공유 메모리에 완전히 로드 가능 (메모리 계층 최적화)
# - 타일 간 독립적 처리로 확장성 확보 (병렬성 극대화)
# - 데이터 재사용 극대화 (같은 데이터를 여러 계산에 활용)

# Tiled Implementation 구성 상수들
alias SIZE_TILED = 8  # 더 큰 행렬 크기 (8×8)
alias BLOCKS_PER_GRID_TILED = (3, 3)  # 3×3 블록 그리드 (각 블록이 3×3 타일 처리)
alias THREADS_PER_BLOCK_TILED = (TPB, TPB)  # 각 블록 내 3×3 스레드
alias layout_tiled = Layout.row_major(SIZE_TILED, SIZE_TILED)  # 8×8 행렬 레이아웃


# Tiled Matrix Multiplication 구현
#
# 타일링 전략 (8×8 행렬을 3×3 타일로 분할):
#
# 행렬 분할 패턴:
# +-------+-------+---+
# | T(0,0)| T(0,1)|T02|  ← 각 타일은 최대 3×3 크기
# +-------+-------+---+
# | T(1,0)| T(1,1)|T12|  ← 경계 타일은 부분적으로 채워짐
# +-------+-------+---+
# | T20   | T21   |T22|  ← 마지막 타일들은 2×2 크기
# +-------+-------+---+
#
# 블록-타일 매핑:
# Block(0,0) → Tile[0:3, 0:3]    Block(0,1) → Tile[0:3, 3:6]    Block(0,2) → Tile[0:3, 6:8]
# Block(1,0) → Tile[3:6, 0:3]    Block(1,1) → Tile[3:6, 3:6]    Block(1,2) → Tile[3:6, 6:8]
# Block(2,0) → Tile[6:8, 0:3]    Block(2,1) → Tile[6:8, 3:6]    Block(2,2) → Tile[6:8, 6:8]
#
# 각 블록이 독립적으로 하나의 출력 타일을 계산
# 타일 간 통신 없이 병렬 처리 가능
fn matmul_tiled[
    layout: Layout, size: Int
](
    output: LayoutTensor[mut=False, dtype, layout],
    a: LayoutTensor[mut=False, dtype, layout],
    b: LayoutTensor[mut=False, dtype, layout],
):
    # 인덱스 계산
    local_row = thread_idx.x  # 타일 내 로컬 행 인덱스
    local_col = thread_idx.y  # 타일 내 로컬 열 인덱스
    tiled_row = block_idx.x * TPB + thread_idx.x  # 전체 행렬에서의 글로벌 행 인덱스
    tiled_col = block_idx.y * TPB + thread_idx.y  # 전체 행렬에서의 글로벌 열 인덱스

    # FILL ME IN (roughly 20 lines)

    # 공유 메모리 할당: 각 타일을 위한 TPB×TPB 크기
    a_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()
    b_shared = tb[dtype]().row_major[TPB, TPB]().shared().alloc()

    # 누적 변수: 최종 결과를 저장할 레지스터 변수
    # 모든 타일의 부분 결과를 누적하여 최종 내적 계산
    var acc: output.element_type = 0

    # 타일별 순차 처리 루프
    #
    # 🔄 중요: 이 루프는 각 블록 내에서 순차적으로 실행되지만,
    #         모든 블록이 동시에 같은 단계를 수행합니다!
    #
    # 타일 개수 계산: (size + TPB - 1) // TPB = (8 + 3 - 1) // 3 = 3개 타일
    #
    # 🎯 각 블록의 역할 (동시 실행):
    # Block(0,0) → 출력 C[0:3, 0:3] 담당
    # Block(0,1) → 출력 C[0:3, 3:6] 담당
    # Block(0,2) → 출력 C[0:3, 6:8] 담당
    # Block(1,0) → 출력 C[3:6, 0:3] 담당
    # Block(1,1) → 출력 C[3:6, 3:6] 담당
    # Block(1,2) → 출력 C[3:6, 6:8] 담당
    # Block(2,0) → 출력 C[6:8, 0:3] 담당
    # Block(2,1) → 출력 C[6:8, 3:6] 담당
    # Block(2,2) → 출력 C[6:8, 6:8] 담당
    #
    # 📊 타일 처리 순서 (Block(1,1) 예시 - 출력 타일 [3:6, 3:6] 계산):
    # Tile 0: A[3:6, 0:3] × B[0:3, 3:6] → 부분 내적 1
    # Tile 1: A[3:6, 3:6] × B[3:6, 3:6] → 부분 내적 2
    # Tile 2: A[3:6, 6:8] × B[6:8, 3:6] → 부분 내적 3
    # 최종 결과 = 부분 내적 1 + 부분 내적 2 + 부분 내적 3
    #
    # 🧮 수학적 배경 - 블록 행렬 곱셈의 수학적 정당성
    #
    # 🔍 핵심 질문: 왜 타일링이 수학적으로 올바른가?
    #
    # 📜 정리 1: 블록 행렬 곱셈 정리
    # 임의의 행렬 A, B를 블록으로 나누어 곱셈할 수 있다.
    #
    # 📝 수학적 증명:
    #
    # 1단계: 기본 설정
    # 8×8 행렬을 다음과 같이 블록으로 분할:
    # A = [A₀₀  A₀₁  A₀₂]     B = [B₀₀  B₀₁  B₀₂]
    #     [A₁₀  A₁₁  A₁₂]         [B₁₀  B₁₁  B₁₂]
    #     [A₂₀  A₂₁  A₂₂]         [B₂₀  B₂₁  B₂₂]
    #
    # 여기서:
    # - A₀₀, B₀₀: 3×3 블록 (완전한 타일)
    # - A₀₂, B₀₂: 3×2 블록 (경계 블록)
    # - A₂₀, B₂₀: 2×3 블록 (경계 블록)
    # - A₂₂, B₂₂: 2×2 블록 (모서리 블록)
    #
    # 2단계: 블록 곱셈 공식
    # 정리: C = A × B에서 각 블록 C_{ij}는 다음과 같이 계산됨:
    # C_{ij} = Σ(k=0 to 2) A_{ik} × B_{kj}
    #
    # 구체적으로:
    # C₁₁ = A₁₀×B₀₁ + A₁₁×B₁₁ + A₁₂×B₂₁
    #
    # 3단계: 원소별 검증
    # 일반적인 행렬 곱셈:
    # C[i,j] = Σ(k=0 to 7) A[i,k] × B[k,j]
    #
    # 블록 행렬 곱셈:
    # C₁₁[1,1] = (A₁₀×B₀₁ + A₁₁×B₁₁ + A₁₂×B₂₁)[1,1]
    #          = (A₁₀×B₀₁)[1,1] + (A₁₁×B₁₁)[1,1] + (A₁₂×B₂₁)[1,1]
    #
    # 각 블록 곱셈을 전개하면:
    # (A₁₀×B₀₁)[1,1] = Σ(k=0 to 2) A₁₀[1,k] × B₀₁[k,1] = Σ(k=0 to 2) A[4,k] × B[k,4]
    # (A₁₁×B₁₁)[1,1] = Σ(k=0 to 2) A₁₁[1,k] × B₁₁[k,1] = Σ(k=3 to 5) A[4,k] × B[k,4]
    # (A₁₂×B₂₁)[1,1] = Σ(k=0 to 1) A₁₂[1,k] × B₂₁[k,1] = Σ(k=6 to 7) A[4,k] × B[k,4]
    #
    # 합치면:
    # C₁₁[1,1] = Σ(k=0 to 2) A[4,k]×B[k,4] + Σ(k=3 to 5) A[4,k]×B[k,4] + Σ(k=6 to 7) A[4,k]×B[k,4]
    #          = Σ(k=0 to 7) A[4,k] × B[k,4] = C[4,4]  ✅
    #
    # 결론: 블록 행렬 곱셈 결과가 일반 행렬 곱셈과 동일함이 증명됨!
    #
    # 🔢 구체적인 숫자 예시로 검증:
    #
    # 원본 8×8 행렬 설정:
    # A = [0  1  2  3  4  5  6  7 ]
    #     [8  9  10 11 12 13 14 15]
    #     [16 17 18 19 20 21 22 23]
    #     [24 25 26 27 28 29 30 31]
    #     [32 33 34 35 36 37 38 39]  ← 4번째 행
    #     [40 41 42 43 44 45 46 47]
    #     [48 49 50 51 52 53 54 55]
    #     [56 57 58 59 60 61 62 63]
    # B = A × 2
    #
    # 일반 행렬 곱셈으로 C[4,4] 계산:
    # A의 4번째 행: [32, 33, 34, 35, 36, 37, 38, 39]
    # B의 4번째 열: [8, 24, 40, 56, 72, 88, 104, 120]
    # C[4,4] = 32×8 + 33×24 + 34×40 + 35×56 + 36×72 + 37×88 + 38×104 + 39×120
    #        = 256 + 792 + 1360 + 1960 + 2592 + 3256 + 3952 + 4680 = 18848
    #
    # 블록 행렬 곱셈으로 타일별 계산:
    # 타일 0 (k=0~2): A[4, 0:3] × B[0:3, 4] = [32,33,34] × [8,24,40]ᵀ = 2408
    # 타일 1 (k=3~5): A[4, 3:6] × B[3:6, 4] = [35,36,37] × [56,72,88]ᵀ = 7808
    # 타일 2 (k=6~7): A[4, 6:8] × B[6:8, 4] = [38,39] × [104,120]ᵀ = 8632
    # 최종: C[4,4] = 2408 + 7808 + 8632 = 18848 ✅
    #
    # 🎯 타일링의 핵심 아이디어:
    # C[i,j] = Σ(k=0 to 7) A[i,k] × B[k,j]
    #        = Σ(k=0 to 2) A[i,k] × B[k,j] + Σ(k=3 to 5) A[i,k] × B[k,j] + Σ(k=6 to 7) A[i,k] × B[k,j]
    #          ↑ Tile 0              ↑ Tile 1              ↑ Tile 2
    #
    # ⚡ 병렬성의 핵심:
    # - 9개 블록이 모두 동시에 이 루프를 실행
    # - 각 블록은 서로 다른 A, B 데이터 영역을 로드
    # - 각 블록은 서로 다른 C 출력 영역에 결과 저장
    # - 블록 간 데이터 의존성 없음 → 완전 병렬 처리 가능
    #
    # 🔬 수학적 원리의 실제 구현:
    # 위의 수학적 증명이 코드에서 어떻게 실현되는지:
    #
    # Block(1,1)이 C[4,4] 계산하는 과정:
    # tile=0: a_shared에 A[3:6,0:3] 로드, b_shared에 B[0:3,3:6] 로드
    #         → acc += A[4,0:3] × B[0:3,4] (수학적 증명의 첫 번째 합)
    # tile=1: a_shared에 A[3:6,3:6] 로드, b_shared에 B[3:6,3:6] 로드
    #         → acc += A[4,3:6] × B[3:6,4] (수학적 증명의 두 번째 합)
    # tile=2: a_shared에 A[3:6,6:8] 로드, b_shared에 B[6:8,3:6] 로드
    #         → acc += A[4,6:8] × B[6:8,4] (수학적 증명의 세 번째 합)
    #
    # 최종 결과: acc = Σ(k=0 to 7) A[4,k] × B[k,4] = C[4,4]
    # 이는 위의 수학적 증명과 정확히 일치! ✅
    @parameter
    for tile in range((size + TPB - 1) // TPB):
        # 공유 메모리 초기화
        #
        # 이전 타일의 데이터를 제거하여 올바른 계산 보장
        # 경계 타일에서 유효하지 않은 데이터로 인한 오류 방지
        # 특히 마지막 타일(6:8 범위)에서 중요
        if local_row < TPB and local_col < TPB:
            a_shared[local_row, local_col] = 0
            b_shared[local_row, local_col] = 0

        barrier()  # 초기화 완료 대기

        # A 타일 로딩: 행은 고정, 열은 타일에 따라 변경
        #
        # 🔍 각 블록별 A 타일 로딩 패턴 (Tile 1 예시):
        # Block(0,0): A[0:3, 3:6] → a_shared[0:3, 0:3]
        # Block(0,1): A[0:3, 3:6] → a_shared[0:3, 0:3] (같은 A 영역!)
        # Block(1,0): A[3:6, 3:6] → a_shared[0:3, 0:3]
        # Block(1,1): A[3:6, 3:6] → a_shared[0:3, 0:3]
        # Block(2,0): A[6:8, 3:6] → a_shared[0:2, 0:3]
        #
        # 📊 세부 스레드 로딩 (Block(1,1), Tile 1 예시):
        # Thread(0,0): A[3, 3] → a_shared[0, 0]
        # Thread(0,1): A[3, 4] → a_shared[0, 1]
        # Thread(1,0): A[4, 3] → a_shared[1, 0]
        # Thread(1,1): A[4, 4] → a_shared[1, 1]
        #
        # 💡 핵심 관찰:
        # - 같은 행의 블록들(Block(0,0), Block(0,1), Block(0,2))은 동일한 A 영역 로드
        # - 하지만 각자의 공유 메모리에 독립적으로 저장
        # - 메모리 코얼레싱: 연속된 열 인덱스로 접근하여 효율적 로딩
        # - 경계 검사: 행렬 범위를 벗어나는 접근 방지
        if tiled_row < size and (tile * TPB + local_col) < size:
            a_shared[local_row, local_col] = a[
                tiled_row, tile * TPB + local_col
            ]

        # B 타일 로딩: 행은 타일에 따라 변경, 열은 고정
        #
        # 🔍 각 블록별 B 타일 로딩 패턴 (Tile 1 예시):
        # Block(0,0): B[3:6, 0:3] → b_shared[0:3, 0:3]
        # Block(0,1): B[3:6, 3:6] → b_shared[0:3, 0:3]
        # Block(0,2): B[3:6, 6:8] → b_shared[0:3, 0:2]
        # Block(1,0): B[3:6, 0:3] → b_shared[0:3, 0:3] (같은 B 영역!)
        # Block(1,1): B[3:6, 3:6] → b_shared[0:3, 0:3]
        # Block(1,2): B[3:6, 6:8] → b_shared[0:3, 0:2]
        #
        # 📊 세부 스레드 로딩 (Block(1,1), Tile 1 예시):
        # Thread(0,0): B[3, 3] → b_shared[0, 0]
        # Thread(0,1): B[3, 4] → b_shared[0, 1]
        # Thread(1,0): B[4, 3] → b_shared[1, 0]
        # Thread(1,1): B[4, 4] → b_shared[1, 1]
        #
        # 💡 핵심 관찰:
        # - 같은 열의 블록들(Block(0,0), Block(1,0), Block(2,0))은 동일한 B 영역 로드
        # - 하지만 각자의 공유 메모리에 독립적으로 저장
        # - 전치 접근 패턴: B 행렬의 열을 행으로 로딩하여 효율적 계산
        if (tile * TPB + local_row) < size and tiled_col < size:
            b_shared[local_row, local_col] = b[
                tile * TPB + local_row, tiled_col
            ]

        barrier()  # 로딩 완료 대기

        # 타일 내 행렬 곱셈 수행
        #
        # 🧮 부분 내적 계산: 현재 타일에서의 기여도만 계산
        #
        # 📊 동시 계산 예시 (모든 블록이 Tile 1 처리 중):
        # Block(0,0): a_shared[0:3,0:3] × b_shared[0:3,0:3] → acc에 누적
        # Block(0,1): a_shared[0:3,0:3] × b_shared[0:3,0:3] → acc에 누적
        # Block(1,1): a_shared[0:3,0:3] × b_shared[0:3,0:3] → acc에 누적
        # ... 9개 블록이 각자의 공유 메모리 데이터로 동시 계산!
        #
        # 🔢 계산 범위 결정: min(TPB, size - tile * TPB)
        # - 완전한 타일 (tile 0, 1): TPB = 3
        # - 경계 타일 (tile 2): size - tile * TPB = 8 - 2*3 = 2
        # - 이는 행렬 경계를 넘지 않도록 하는 안전장치
        #
        # ⚡ 병렬성 강조:
        # - 각 블록의 acc 변수는 독립적 (레지스터에 저장)
        # - 9개 블록이 동시에 서로 다른 부분 내적 계산
        # - 블록 간 간섭 없이 완전 병렬 수행
        if tiled_row < size and tiled_col < size:

            @parameter
            for k in range(min(TPB, size - tile * TPB)):
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()  # 계산 완료 대기 (다음 타일 처리 전 동기화)

    # 모든 타일의 부분 결과를 누적한 최종 값을 출력 행렬에 저장
    # 경계 검사로 유효한 결과만 저장
    if tiled_row < size and tiled_col < size:
        output[tiled_row, tiled_col] = acc


# ANCHOR_END: matmul_tiled


def main():
    with DeviceContext() as ctx:
        # 실행 모드에 따른 구성 선택
        #
        # 세 가지 구현 방법 비교:
        # 1. --naive: 기본 구현 (2×2 행렬, 단순한 접근법)
        # 2. --single-block: 공유 메모리 사용 (2×2 행렬, 메모리 최적화)
        # 3. --tiled: 타일링 기법 (8×8 행렬, 확장성 있는 접근법)
        size = SIZE_TILED if argv()[1] == "--tiled" else SIZE

        # GPU 메모리 버퍼 생성
        out = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp1 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        inp2 = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](
            size * size
        ).enqueue_fill(0)

        # 입력 데이터 초기화
        #
        # 행렬 A: [0, 1, 2, 3, ...]  (순차적 증가)
        # 행렬 B: [0, 2, 4, 6, ...]  (A의 2배)
        #
        # 이 패턴은 계산 결과 검증을 용이하게 함
        with inp1.map_to_host() as inp1_host, inp2.map_to_host() as inp2_host:
            for row in range(size):
                for col in range(size):
                    val = row * size + col
                    # row major: 행별로 원소 배치
                    inp1_host[row * size + col] = val
                    inp2_host[row * size + col] = Float32(2.0) * val

            # CPU 참조 구현: inp1 @ inp2 (표준 행렬 곱셈)
            #
            # 삼중 루프를 사용한 전통적인 행렬 곱셈 알고리즘
            # GPU 결과와 비교하여 정확성 검증
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        expected[i * size + j] += (
                            inp1_host[i * size + k] * inp2_host[k * size + j]
                        )

        # LayoutTensor 생성: 1D 버퍼를 2D 행렬로 해석
        out_tensor = LayoutTensor[mut=False, dtype, layout](out.unsafe_ptr())
        a_tensor = LayoutTensor[mut=False, dtype, layout](inp1.unsafe_ptr())
        b_tensor = LayoutTensor[mut=False, dtype, layout](inp2.unsafe_ptr())

        # 구현 방법별 GPU 커널 실행
        if argv()[1] == "--naive":
            # Naive 구현 실행
            #
            # 특징:
            # - 가장 간단한 구현
            # - 각 스레드가 하나의 출력 원소 계산
            # - 글로벌 메모리에서 직접 읽기
            # - 메모리 효율성 낮음
            #
            # 성능 특성:
            # - Arithmetic Intensity: 0.15 FLOP/B
            # - 메모리 바운드 연산
            # - GPU 성능의 1.2%만 활용
            ctx.enqueue_function[naive_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,  # (1, 1) - 단일 블록
                block_dim=THREADS_PER_BLOCK,  # (3, 3) - 2D 스레드 블록
            )
        elif argv()[1] == "--single-block":
            # Shared Memory 구현 실행
            #
            # 특징:
            # - 공유 메모리를 활용한 최적화
            # - 협력적 데이터 로딩
            # - 글로벌 메모리 접근 감소
            # - 메모리 계층 구조 활용
            #
            # 성능 특성:
            # - Arithmetic Intensity 증가 (67% 개선)
            # - 메모리 접근 레이턴시 400배 감소
            # - 여전히 메모리 바운드이지만 성능 향상
            ctx.enqueue_function[single_block_matmul[layout, SIZE]](
                out_tensor,
                a_tensor,
                b_tensor,
                grid_dim=BLOCKS_PER_GRID,  # (1, 1) - 단일 블록
                block_dim=THREADS_PER_BLOCK,  # (3, 3) - 2D 스레드 블록
            )
        elif argv()[1] == "--tiled":
            # Tiled 구현 실행
            #
            # 특징:
            # - 대용량 행렬 처리 가능
            # - 다중 블록 활용
            # - 타일별 순차 처리
            # - 메모리 사용량 최적화
            #
            # 성능 특성:
            # - 가장 높은 Arithmetic Intensity (8.3배 개선)
            # - 확장성 있는 설계
            # - 실제 GPU 라이브러리에서 사용하는 방법
            # - 큰 행렬에서 compute-bound 달성 가능

            # 더 큰 행렬을 위한 레이아웃 업데이트 필요
            out_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                out.unsafe_ptr()
            )
            a_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp1.unsafe_ptr()
            )
            b_tensor_tiled = LayoutTensor[mut=False, dtype, layout_tiled](
                inp2.unsafe_ptr()
            )

            ctx.enqueue_function[matmul_tiled[layout_tiled, SIZE_TILED]](
                out_tensor_tiled,
                a_tensor_tiled,
                b_tensor_tiled,
                grid_dim=BLOCKS_PER_GRID_TILED,  # (3, 3) - 다중 블록
                block_dim=THREADS_PER_BLOCK_TILED,  # (3, 3) - 2D 스레드 블록
            )
        else:
            raise Error("Invalid argument")

        ctx.synchronize()

        # 결과 검증 및 성능 분석
        #
        # 세 가지 구현 방법의 Roofline Model 기반 성능 비교:
        #
        # 1. Naive Implementation:
        #    - Arithmetic Intensity: 0.15 FLOP/B
        #    - 예상 성능: ~233 GFLOP/s (GPU 피크의 1.2%)
        #    - 장점: 구현 단순, 이해하기 쉬움
        #    - 단점: 메모리 효율성 극히 낮음, 확장성 없음
        #    - Roofline 위치: 메모리 루프 최하단 (심각한 메모리 바운드)
        #    - 용도: 학습용, 프로토타입
        #
        # 2. Shared Memory Implementation:
        #    - Arithmetic Intensity: 0.25 FLOP/B (67% 개선)
        #    - 예상 성능: ~389 GFLOP/s (GPU 피크의 2.0%)
        #    - 장점: 메모리 접근 최적화, 데이터 재사용
        #    - 단점: 단일 블록 제한, 큰 행렬 처리 불가
        #    - Roofline 위치: 메모리 루프 중간 (개선된 메모리 바운드)
        #    - 용도: 작은 행렬, 메모리 최적화 학습
        #
        # 3. Tiled Implementation:
        #    - Arithmetic Intensity: 1.25 FLOP/B (8.3배 개선)
        #    - 예상 성능: ~1,944 GFLOP/s (GPU 피크의 10.0%)
        #    - 장점: 확장성, 대용량 처리, 실용적 성능
        #    - 단점: 구현 복잡도 높음, 디버깅 어려움
        #    - Roofline 위치: 메모리 루프 상단 (임계점 근접)
        #    - 용도: 실제 애플리케이션, 고성능 라이브러리
        #
        # Roofline Model에서의 최적화 여정:
        # Naive (0.15) → Shared (0.25) → Tiled (1.25) FLOP/B
        # 메모리 루프를 따라 상승하며 컴퓨트 루프에 근접
        #
        # 더 큰 행렬에서의 확장성:
        # - 512×512: I ≈ 85 FLOP/B (컴퓨트 바운드 달성)
        # - 1024×1024: I ≈ 341 FLOP/B (GPU 피크 성능 근접)
        # - 실제 딥러닝: 수천×수천 행렬에서 최적 성능
        with out.map_to_host() as out_host:
            print("out:", out_host)  # GPU 계산 결과
            print("expected:", expected)  # CPU 참조 결과
            for col in range(size):
                for row in range(size):
                    assert_equal(
                        out_host[col * size + row], expected[col * size + row]
                    )
