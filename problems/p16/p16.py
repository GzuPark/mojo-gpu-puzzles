# ANCHOR: softmax_custom_op_graph
from pathlib import Path
from time import perf_counter
import numpy as np
from max.driver import CPU, Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray
from scipy.special import softmax as scipy_softmax

# Batched Softmax 연산을 위한 MAX Graph 통합
# Row-wise vs Column-wise 확률 분포 정규화
#
# Batched Softmax 함수의 특징:
# 1. 2D 입력 텐서 (BATCH_SIZE, FEATURE_SIZE)
# 2. Row-wise: 각 행(row)에 대해 독립적으로 softmax 적용
# 3. Column-wise: 각 열(column)에 대해 독립적으로 softmax 적용
# 4. 성능 비교: 메모리 접근 패턴과 병렬성의 트레이드오프

def compile_softmax_model(
    input_shape: tuple,
    session: InferenceSession,
    device: Device,
    mode: str = "row_wise",  # "row_wise" 또는 "col_wise"
):
    """Softmax 모델을 컴파일하고 반환합니다. (한 번만 컴파일)"""
    dtype = DType.float32
    mojo_kernels = Path(__file__).parent / "op"

    with Graph(
        f"softmax_graph_{mode}",
        input_types=[
            TensorType(
                dtype,
                shape=input_shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        input_value = graph.inputs[0]

        # Batched Softmax는 입력과 동일한 크기의 출력 생성
        output = ops.custom(
            name="softmax",
            values=[input_value],
            device=DeviceRef.from_device(device),
            out_types=[
                TensorType(
                    dtype=input_value.tensor.dtype,
                    shape=input_value.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={
                "batch_size": input_shape[0],  # 배치 크기 (행 개수)
                "feature_size": input_shape[1],  # 특성 크기 (열 개수)
                "mode": mode,  # "row_wise" 또는 "col_wise"
                "dtype": dtype,
                "target": "gpu" if device == Accelerator() else "cpu",
            },
        )[0].tensor
        graph.output(output)

    print(f"📦 {mode} softmax 모델을 {device}에서 컴파일 중...")
    compiled_model = session.load(graph)
    print(f"✅ {mode} 모델 컴파일 완료")
    return compiled_model


def execute_softmax_model(
    compiled_model,
    input_array: NDArray[np.float32],
    device: Device,
) -> Tensor:
    """컴파일된 모델로 softmax를 실행합니다."""
    input_tensor = Tensor.from_numpy(input_array).to(device)
    result = compiled_model.execute(input_tensor)[0]
    assert isinstance(result, Tensor)
    return result.to(CPU()) if device == Accelerator() else result


def benchmark_softmax_optimized(
    input_array: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
    warmup_runs: int = 5,
    benchmark_runs: int = 20
) -> dict:
    """컴파일과 실행을 분리한 최적화된 벤치마크"""

    results = {}
    compiled_models = {}

    print(f"🔧 컴파일 단계")
    print(f"{'='*50}")

    # 1단계: 모든 모드에 대해 사전 컴파일
    for mode in ["row_wise", "col_wise"]:
        compiled_models[mode] = compile_softmax_model(
            input_array.shape, session, device, mode
        )

    print(f"\n🚀 실행 단계")
    print(f"{'='*50}")

    # 2단계: 컴파일된 모델로 순수 실행 시간 측정
    for mode in ["row_wise", "col_wise"]:
        mode_kr = "행별" if mode == "row_wise" else "열별"
        print(f"\n=== {mode_kr.upper()} SOFTMAX 벤치마크 ===")

        compiled_model = compiled_models[mode]

        # Warmup 실행 (GPU 메모리 최적화)
        print(f"🔥 {mode_kr} 실행 워밍업 중...")
        for i in range(warmup_runs):
            _ = execute_softmax_model(compiled_model, input_array, device)
            if i % 2 == 0:
                print(f"  워밍업 {i+1}/{warmup_runs}")

        # 순수 실행 시간 측정 (컴파일 제외)
        print(f"⏱️  순수 실행 시간 측정 중...")
        times = []
        for i in range(benchmark_runs):
            start_time = perf_counter()
            result = execute_softmax_model(compiled_model, input_array, device)
            end_time = perf_counter()

            execution_time = (end_time - start_time) * 1000  # ms 단위
            times.append(execution_time)

            if i % 5 == 0 or i == benchmark_runs - 1:
                print(f"  실행 {i+1}/{benchmark_runs}: {execution_time:.3f} ms")

        # 통계 계산
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        median_time = np.median(times)

        results[mode] = {
            "times": times,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "median_time": median_time,
            "result": result
        }

        print(f"📊 결과:")
        print(f"  평균:   {avg_time:.3f} ms")
        print(f"  중간값: {median_time:.3f} ms")
        print(f"  최소:   {min_time:.3f} ms")
        print(f"  최대:   {max_time:.3f} ms")
        print(f"  표준편차: {std_time:.3f} ms")

    return results


def verify_correctness(results: dict, input_array: NDArray[np.float32]):
    """두 방식의 결과가 올바른지 검증합니다."""

    print(f"\n=== 정확성 검증 ===")

    row_result = results["row_wise"]["result"].to_numpy()
    col_result = results["col_wise"]["result"].to_numpy()

    # 결과 shape 및 기본 정보 출력
    print(f"📐 결과 형태 정보:")
    print(f"  입력 형태: {input_array.shape}")
    print(f"  행별 출력 형태: {row_result.shape}")
    print(f"  열별 출력 형태: {col_result.shape}")
    print(f"  결과당 메모리 사용량: {row_result.nbytes / 1024:.1f} KB")

    # 첫 몇 개 값 출력
    print(f"\n🔍 샘플 결과 (첫 번째 행):")
    print(f"  입력값: {input_array[0][:5]}")
    print(f"  행별 softmax: {row_result[0][:5]}")
    print(f"  열별 softmax: {col_result[0][:5]}")

    # Row-wise: SciPy와 비교 (각 행별 softmax)
    expected_row = np.array([scipy_softmax(row) for row in input_array])
    np.testing.assert_allclose(row_result, expected_row, rtol=1e-5)
    print(f"\n✅ 행별 결과가 SciPy 계산과 일치함")

    # Column-wise: SciPy와 비교 (각 열별 softmax)
    expected_col = np.array([scipy_softmax(col) for col in input_array.T]).T
    np.testing.assert_allclose(col_result, expected_col, rtol=1e-5)
    print("✅ 열별 결과가 SciPy 계산과 일치함")

    # 확률 분포 속성 검증
    row_sums = np.sum(row_result, axis=1)  # 각 행의 합
    col_sums = np.sum(col_result, axis=0)  # 각 열의 합

    assert np.allclose(row_sums, 1.0, atol=1e-6), f"행별 합이 1.0이 아님: {row_sums}"
    assert np.allclose(col_sums, 1.0, atol=1e-6), f"열별 합이 1.0이 아님: {col_sums}"

    print(f"✅ 행별 확률 합계: {np.round(row_sums[:5], 5)}")
    print(f"✅ 열별 확률 합계: {np.round(col_sums[:5], 5)}")


def analyze_performance_detailed(results: dict):
    """상세한 성능 분석 및 메모리 접근 패턴 비교"""

    print(f"\n=== 상세 성능 분석 ===")

    row_stats = results["row_wise"]
    col_stats = results["col_wise"]

    row_avg = row_stats["avg_time"]
    col_avg = col_stats["avg_time"]
    row_min = row_stats["min_time"]
    col_min = col_stats["min_time"]
    row_median = row_stats["median_time"]
    col_median = col_stats["median_time"]

    print(f"\n📈 실행 시간 비교:")
    print(f"┌─────────────┬──────────────┬──────────────┬──────────────┐")
    print(f"│ 지표        │ 행별         │ 열별         │ 차이         │")
    print(f"├─────────────┼──────────────┼──────────────┼──────────────┤")
    print(f"│ 평균        │ {row_avg:8.3f} ms  │ {col_avg:8.3f} ms  │ {abs(row_avg-col_avg):8.3f} ms  │")
    print(f"│ 중간값      │ {row_median:8.3f} ms  │ {col_median:8.3f} ms  │ {abs(row_median-col_median):8.3f} ms  │")
    print(f"│ 최고(최소)  │ {row_min:8.3f} ms  │ {col_min:8.3f} ms  │ {abs(row_min-col_min):8.3f} ms  │")
    print(f"└─────────────┴──────────────┴──────────────┴──────────────┘")

    if row_avg < col_avg:
        speedup = col_avg / row_avg
        winner = "행별"
        print(f"\n🏆 {winner} 방식이 {speedup:.2f}x 빠름!")
    else:
        speedup = row_avg / col_avg
        winner = "열별"
        print(f"\n🏆 {winner} 방식이 {speedup:.2f}x 빠름!")

    print(f"\n🧠 메모리 접근 패턴 분석:")
    print(f"")
    print(f"행별 (순차 접근 패턴):")
    print(f"  🟢 캐시 지역성: 우수 (연속 메모리 접근)")
    print(f"  🟢 메모리 대역폭: 높은 활용률 (~80-90%)")
    print(f"  🟢 DRAM 버스트 효율성: 최적 (128바이트 캐시 라인)")
    print(f"  🔴 병렬성: 배치 차원에 의해 제한됨")
    print(f"  📊 최적화 대상: 큰 특성 크기, 작은 배치 크기")

    print(f"")
    print(f"열별 (스트라이드 접근 패턴):")
    print(f"  🔴 캐시 지역성: 불량 (스트라이드 메모리 접근)")
    print(f"  🔴 메모리 대역폭: 낮은 활용률 (~30-50%)")
    print(f"  🔴 DRAM 버스트 효율성: 비최적 (많은 캐시 미스)")
    print(f"  🟢 병렬성: 특성 차원에서 최대")
    print(f"  📊 최적화 대상: 작은 특성 크기, 큰 배치 크기")

    # 성능 변동성 분석
    row_cv = (row_stats["std_time"] / row_stats["avg_time"]) * 100
    col_cv = (col_stats["std_time"] / col_stats["avg_time"]) * 100

    print(f"\n📊 성능 안정성:")
    print(f"행별 변동계수: {row_cv:.2f}%")
    print(f"열별 변동계수: {col_cv:.2f}%")

    if row_cv < col_cv:
        print(f"✅ 행별이 더 일관된 성능을 보임")
    else:
        print(f"✅ 열별이 더 일관된 성능을 보임")


if __name__ == "__main__":
    # 2D Batched Softmax 테스트 설정
    BATCH_SIZE = 8      # 배치 크기 증가 (더 현실적인 테스트)
    FEATURE_SIZE = 512  # 특성 크기 증가 (메모리 패턴 차이 극대화)

    cpu_session = InferenceSession(devices=[CPU()])
    gpu_session = InferenceSession(devices=[Accelerator()])

    # 2D 입력 배열 생성 (BATCH_SIZE, FEATURE_SIZE)
    np.random.seed(42)  # 재현 가능한 결과를 위한 시드 설정
    input_array = np.random.randn(BATCH_SIZE, FEATURE_SIZE).astype(np.float32)

    print(f"🎯 종합 Batched Softmax 벤치마크")
    print(f"{'='*80}")
    print(f"입력 형태: {input_array.shape}")
    print(f"배치 크기: {BATCH_SIZE}, 특성 크기: {FEATURE_SIZE}")
    print(f"총 원소 수: {BATCH_SIZE * FEATURE_SIZE:,}개")
    print(f"메모리 사용량: {input_array.nbytes / 1024:.1f} KB")
    print(f"데이터 타입: {input_array.dtype}")
    print(f"첫 번째 행 입력값 샘플: {input_array[0][:5]}")

    # CPU 벤치마크 실행
    print(f"\n💻 CPU 벤치마크")
    print(f"{'='*50}")
    cpu_results = benchmark_softmax_optimized(
        input_array, cpu_session, CPU(),
        warmup_runs=3, benchmark_runs=20  # CPU는 GPU보다 적은 실행
    )

    # GPU 벤치마크 실행 (컴파일 분리)
    print(f"\n🔥 GPU 벤치마크")
    print(f"{'='*50}")
    gpu_results = benchmark_softmax_optimized(
        input_array, gpu_session, Accelerator(),
        warmup_runs=5, benchmark_runs=50  # 더 정확한 측정을 위해 실행 횟수 증가
    )

    verify_correctness(gpu_results, input_array)
    analyze_performance_detailed(gpu_results)
