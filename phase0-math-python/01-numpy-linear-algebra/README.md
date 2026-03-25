# 📐 Phase 0 - 01: NumPy Linear Algebra Basics

> **Phase 0 — 수학 & Python 기초** | 구현 과제 1 / 3

[![wandb](https://img.shields.io/badge/W%26B-실험%20보기-yellow?logo=weightsandbiases)](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab/workspace?nw=nwuserswsw778)
[![Colab](https://img.shields.io/badge/Colab-노트북%20열기-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/Wo0ond3r/portfolio-tsuruoka-lab/blob/main/phase0-math-python/01-numpy-linear-algebra/01_numpy_linear_algebra_basics.ipynb)

---

## 개요

행렬 곱셈과 브로드캐스팅의 **수식 → 코드 → 직관** 연결.
이 두 연산은 Transformer의 모든 선형 변환과 Attention 마스킹의 기반입니다.

---

## 구현 내용

### 1. 행렬 곱셈 — 3가지 방법

$$C_{ij} = \sum_{k} A_{ik} \cdot B_{kj}$$

| 방법     | 코드                                 | 용도                  |
| -------- | ------------------------------------ | --------------------- |
| for loop | `for i,j,k: C[i,j] += A[i,k]*B[k,j]` | 원리 이해             |
| NumPy    | `A @ B`                              | 실전 사용             |
| einsum   | `np.einsum('ik,kj->ij', A, B)`       | Transformer 코드 독해 |

### 2. 속도 비교 실험

| 행렬 크기 | for loop | NumPy @ | 속도 향상 |
| --------- | -------- | ------- | --------- |
| 16×16     | ~X ms    | ~Y ms   | ~Zx       |
| 32×32     | ~X ms    | ~Y ms   | ~Zx       |
| 64×64     | ~X ms    | ~Y ms   | ~Zx       |

> 📊 **정확한 수치는 [W&B 실험 보기](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab/workspace?nw=nwuserswsw778)**

### 3. 브로드캐스팅 — 4가지 핵심 케이스

```python
# 케이스 1: 행 방향 확장  (Dense 레이어 편향)
(32, 64) + (64,)        → (32, 64)

# 케이스 2: 열 방향 확장  (샘플별 스케일)
(32, 64) + (32, 1)      → (32, 64)

# 케이스 3: 신경망 한 층  y = x @ W + b
(32,128) @ (128,64) + (64,)  → (32, 64)

# 케이스 4: Transformer Attention 마스크
(2,4,10,10) + (10,10)   → (2,4,10,10)
```

**규칙:** 오른쪽부터 비교 — 같거나 둘 중 하나가 1이면 OK.
차원 수가 부족하면 앞에 1을 채워서 비교.

### 4. 코사인 유사도

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

단일 벡터 쌍 → N×N 행렬 연산으로 확장 구현.
나중에 단어 임베딩 유사도 계산에서 그대로 재사용.

---

## 실험 결과

### 속도 비교 차트

<!-- W&B 차트를 스크린샷으로 저장 후 아래 경로에 추가 -->

![속도 비교](../assets/matmul_speed_comparison.png)

### 코사인 유사도 히트맵

![코사인 유사도](../assets/cosine_similarity_heatmap.png)

---

## AI와의 연결

| 오늘 구현한 것                   | Phase 2+ 에서 만나는 곳              |
| -------------------------------- | ------------------------------------ |
| `A @ B`                          | Transformer 선형 변환 전체           |
| `einsum('bhqd,bhkd->bhqk')`      | Multi-head Attention score           |
| `(B,H,T,T) + (T,T)` 브로드캐스팅 | Causal attention mask                |
| 코사인 유사도 행렬               | Word embedding 유사도, RAG retrieval |

---

## 파일 구조

```
01-numpy-linear-algebra/
├── notebook/
│   └── linear_algebra_basics.ipynb   ← 메인 노트북 (Colab)
├── src/
│   └── linalg_utils.py               ← 재사용 가능한 함수 모음
├── docs/
│   └── 선형대수_수식_정리.md           ← 수식 유도 & 직관 설명
├── assets/
│   ├── matmul_speed_comparison.png
│   └── cosine_similarity_heatmap.png
├── requirements.txt
└── README.md                          ← 이 파일
```

---

## 배운 것 한 줄 요약

> `C[i][j]` = A의 i행과 B의 j열의 내적. 이 연산이 GPU에서 수백만 번 동시에 일어나는 것이 딥러닝의 엔진이다.

---

## 다음 과제

**[→ 구현 과제 2: Micrograd — 자동미분 엔진 직접 구현](../02-micrograd-from-scratch/)**
