from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray


def attention(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    session: InferenceSession,
    device: Device,
) -> Tensor:
    """
    Compute vector attention: Attention(Q, K, V) = softmax(Q · K^T) @ V

    Args:
        q: Query vector of shape (d,)
        k: Key matrix of shape (seq_len, d)
        v: Value matrix of shape (seq_len, d)
        session: MAX inference session
        device: Target device (CPU or GPU)

    Returns:
        Attention output vector of shape (d,)
    """
    dtype = DType.float32
    seq_len, d = k.shape

    # 입력을 MAX 텐서로 변환하여 지정된 장치로 이동
    q_tensor = Tensor.from_numpy(q).to(device)
    k_tensor = Tensor.from_numpy(k).to(device)
    v_tensor = Tensor.from_numpy(v).to(device)

    mojo_kernels = Path(__file__).parent / "op"

    # MAX Graph 구성: 어텐션 커스텀 연산 정의
    with Graph(
        "attention_graph",
        input_types=[
            TensorType(
                dtype,
                shape=q_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=k_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=v_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        q_value = graph.inputs[0]
        k_value = graph.inputs[1]
        v_value = graph.inputs[2]

        # 커스텀 어텐션 연산 호출
        output = ops.custom(
            name="attention",
            values=[q_value, k_value, v_value],
            device=DeviceRef.from_device(device),
            out_types=[
                TensorType(
                    dtype=dtype,
                    shape=(d,),
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={
                "seq_len": seq_len,
                "d": d,
                "dtype": dtype,
            },
        )[0].tensor
        graph.output(output)

    print(f"Compiling attention graph on {device}")
    model = session.load(graph)
    print(f"Executing attention on {device}")
    print("="*100)
    result = model.execute(q_tensor, k_tensor, v_tensor)[0]
    assert isinstance(result, Tensor)
    return result.to(CPU()) if device == Accelerator() else result


def reference_attention(q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32]) -> NDArray[np.float32]:
    """NumPy를 사용한 벡터 어텐션 참조 구현."""
    scores = np.dot(k, q)  # 각 키와 쿼리의 유사도 점수
    scores_max = np.max(scores)  # 수치적 안정성을 위한 최댓값
    scores_exp = np.exp(scores - scores_max)  # 안전한 지수 계산
    attention_weights = scores_exp / np.sum(scores_exp)  # 정규화된 가중치
    output = np.dot(attention_weights, v)  # 가중 평균 계산
    return output


def debug_attention_steps(q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32]):
    """벡터 어텐션 계산을 단계별로 디버깅하는 함수."""
    print("\n" + "="*80)
    print("STEP-BY-STEP VECTOR ATTENTION COMPUTATION DEBUG")
    print("="*80)

    seq_len, d = k.shape
    print(f"\n1. INPUT SHAPES:")
    print(f"   Q shape: {q.shape} (query vector)")
    print(f"   K shape: {k.shape} (key matrix)")
    print(f"   V shape: {v.shape} (value matrix)")
    print(f"   Q[:5]: {q[:5]}")

    # 1단계: 어텐션 점수 계산 (K[i] · Q)
    scores = np.dot(k, q)  # 각 키 행벡터와 쿼리의 내적
    print(f"\n2. ATTENTION SCORES (K[i] · Q):")
    print(f"   Scores shape: {scores.shape}")
    print(f"   Scores[:5]: {scores[:5]}")
    print(f"   Min: {np.min(scores):.6f}, Max: {np.max(scores):.6f}")

    # 수동 검증: 처음 몇 개 점수 확인
    print(f"   Manual verification:")
    for i in range(min(3, seq_len)):
        manual_score = np.dot(q, k[i])  # Q · K[i] = K[i] · Q
        print(f"     Q · K[{i}] = K[{i}] · Q = {manual_score:.6f} (computed: {scores[i]:.6f})")

    # 2단계: Softmax로 확률 분포 생성
    scores_max = np.max(scores)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / np.sum(scores_exp)
    print(f"\n3. SOFTMAX:")
    print(f"   Max score: {scores_max:.6f}")
    print(f"   Attention weights shape: {attention_weights.shape}")
    print(f"   Attention weights[:5]: {attention_weights[:5]}")
    print(f"   Sum: {np.sum(attention_weights):.6f} (should be 1.0)")

    # 3단계: 값 벡터들의 가중 합
    output = np.dot(attention_weights, v)
    print(f"\n4. WEIGHTED SUM OF VALUES:")
    print(f"   Output shape: {output.shape}")
    print(f"   Output[:5]: {output[:5]}")
    print(f"   Output norm: {np.linalg.norm(output):.6f}")

    # 수동 검증: 가중 합 직접 계산
    manual_output = np.zeros(d, dtype=np.float32)
    for i in range(seq_len):
        manual_output += attention_weights[i] * v[i]
    print(f"   Manual output[:5]: {manual_output[:5]}")
    print(f"   Match: {np.allclose(output, manual_output)}")

    return {
        'scores': scores,
        'attention_weights': attention_weights,
        'output': output
    }


def test_individual_operations():
    """Test individual operations to verify correctness."""
    print(f"\n{'='*80}")
    print(f"TESTING INDIVIDUAL OPERATIONS")
    print(f"{'='*80}")

    # Test 1: Vector dot product
    print("\nTest 1: Vector Dot Product")
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    expected_dot = np.dot(a, b)
    print(f"a · b = {expected_dot:.6f}")

    # Test 2: Matrix-vector multiplication
    print("\nTest 2: Matrix-Vector Multiplication")
    M = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]], dtype=np.float32)
    v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    expected_mv = np.dot(M, v)
    print(f"M @ v = {expected_mv}")

    # Test 3: Softmax
    print("\nTest 3: Softmax")
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    x_softmax = x_exp / np.sum(x_exp)
    print(f"Input: {x}")
    print(f"Softmax: {x_softmax}")
    print(f"Sum: {np.sum(x_softmax):.6f}")


if __name__ == "__main__":
    SEQ_LEN = 16  # Number of key/value vectors
    D = 16  # Dimension of each vector

    cpu_session = InferenceSession(devices=[CPU()])
    gpu_session = InferenceSession(devices=[Accelerator()])

    # 재현 가능한 랜덤 데이터 생성
    np.random.seed(42)
    q = np.random.randn(D).astype(np.float32) * 0.1  # 쿼리 벡터
    k = np.random.randn(SEQ_LEN, D).astype(np.float32) * 0.1  # 키 매트릭스
    v = np.random.randn(SEQ_LEN, D).astype(np.float32) * 0.1  # 값 매트릭스

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Sample Q values: {q[:5]}")
    print(f"Sample K[0] values: {k[0, :5]}")
    print(f"Sample V[0] values: {v[0, :5]}")

    # 단계별 계산 과정 디버깅
    numpy_steps = debug_attention_steps(q, k, v)
    expected_result = numpy_steps['output']

    # 개별 연산 테스트
    test_individual_operations()

    # CPU 및 GPU 구현 테스트
    print(f"\n{'='*80}")
    print("TESTING FULL ATTENTION")
    print(f"{'='*80}")

    # CPU 버전 테스트
    cpu_result = attention(q, k, v, cpu_session, CPU())
    cpu_array = cpu_result.to_numpy()
    print(f"\nCPU attention output[:5]: {cpu_array[:5]}")
    print(f"CPU matches NumPy: {np.allclose(cpu_array, expected_result, rtol=1e-4, atol=1e-4)}")
    if not np.allclose(cpu_array, expected_result, rtol=1e-4, atol=1e-4):
        diff = np.abs(cpu_array - expected_result)
        print(f"Max CPU diff: {np.max(diff):.6f}")
        print(f"CPU diff[:5]: {diff[:5]}")

    # GPU 버전 테스트
    gpu_result = attention(q, k, v, gpu_session, Accelerator())
    gpu_array = gpu_result.to_numpy()
    print(f"\nGPU attention output[:5]: {gpu_array[:5]}")
    print(f"Expected output[:5]: {expected_result[:5]}")
    print(f"GPU matches NumPy: {np.allclose(gpu_array, expected_result, rtol=1e-4, atol=1e-4)}")
    if not np.allclose(gpu_array, expected_result, rtol=1e-4, atol=1e-4):
        diff = np.abs(gpu_array - expected_result)
        print(f"Max GPU diff: {np.max(diff):.6f}")
        print(f"GPU diff[:5]: {diff[:5]}")

        # 가장 큰 차이가 나는 위치 찾기
        max_diff_idx = np.argmax(diff)
        print(f"\nLargest difference at position {max_diff_idx}:")
        print(f"  GPU: {gpu_array[max_diff_idx]:.6f}")
        print(f"  Expected: {expected_result[max_diff_idx]:.6f}")
        print(f"  Diff: {diff[max_diff_idx]:.6f}")

    # 최종 검증
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION")
    print(f"{'='*80}")

    try:
        np.testing.assert_allclose(cpu_array, expected_result, rtol=1e-4, atol=1e-4)
        print("✓ CPU implementation PASSED")
    except AssertionError as e:
        print("✗ CPU implementation FAILED")
        print(str(e))

    try:
        np.testing.assert_allclose(gpu_array, expected_result, rtol=1e-4, atol=1e-4)
        print("✓ GPU implementation PASSED")
    except AssertionError as e:
        print("✗ GPU implementation FAILED")
        print(str(e))

    print(f"\nOutput vector norms:")
    print(f"  CPU: {np.linalg.norm(cpu_array):.6f}")
    print(f"  GPU: {np.linalg.norm(gpu_array):.6f}")
    print(f"  Expected: {np.linalg.norm(expected_result):.6f}")
