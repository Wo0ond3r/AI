"""
linalg_utils.py
---------------
Phase 0 - 구현 과제 1에서 만든 함수들.
이후 과제에서 import해서 재사용합니다.

사용법:
    from src.linalg_utils import cosine_similarity_matrix, matmul_loop
"""

import numpy as np


def matmul_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    행렬 곱셈 — for loop 구현 (원리 이해용).

    C[i][j] = sum_k A[i][k] * B[k][j]

    Args:
        A: shape (m, k)
        B: shape (k, n)
    Returns:
        C: shape (m, n)
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"Shape 불일치: A 열({k}) ≠ B 행({k2})"

    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[p, j]
    return C


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 벡터 a, b의 코사인 유사도.

    cos(a, b) = (a · b) / (||a|| * ||b||)

    Returns:
        float: -1 (반대) ~ 0 (직교) ~ 1 (동일 방향)
    """
    dot    = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b + 1e-8))


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    N개 벡터 간 모든 쌍의 코사인 유사도를 행렬 연산으로 한 번에 계산.

    핵심: 정규화(unit vector) 후 행렬 곱셈 = 모든 쌍의 내적

    Args:
        X: shape (N, D) — N개의 D차원 벡터
    Returns:
        sim: shape (N, N) — 유사도 행렬, sim[i][j] = cos(X[i], X[j])
    """
    # 각 벡터의 L2 norm: (N,1) — keepdims로 브로드캐스팅 준비
    norms = np.linalg.norm(X, axis=1, keepdims=True)   # (N, 1)
    X_norm = X / (norms + 1e-8)                         # (N, D) 브로드캐스팅

    # (N, D) @ (D, N) = (N, N) — 모든 쌍의 내적
    return X_norm @ X_norm.T


def broadcast_check(shape_a: tuple, shape_b: tuple) -> tuple | None:
    """
    두 shape이 브로드캐스팅 가능한지 확인하고 결과 shape을 반환.

    규칙: 오른쪽부터 비교, 같거나 하나가 1이면 OK.
    차원 수 부족 시 앞에 1을 채워서 비교.

    Args:
        shape_a, shape_b: 비교할 두 shape (tuple)
    Returns:
        결과 shape (가능한 경우) 또는 None (불가능한 경우)

    Examples:
        >>> broadcast_check((2,3), (3,))   → (2, 3)
        >>> broadcast_check((2,3), (2,))   → None  (에러)
        >>> broadcast_check((2,3), (2,1))  → (2, 3)
    """
    # 차원 수 맞추기 (앞에 1 채우기)
    ndim = max(len(shape_a), len(shape_b))
    a = (1,) * (ndim - len(shape_a)) + tuple(shape_a)
    b = (1,) * (ndim - len(shape_b)) + tuple(shape_b)

    result = []
    for da, db in zip(a, b):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            return None   # 브로드캐스팅 불가
    return tuple(result)
