# 📐 Phase 0 — 수학 & Python 기초

> **상태: ✅ 완료**  |  기간: 약 3주  |  구현 과제: 3개

[![wandb](https://img.shields.io/badge/W%26B-Phase0%20실험-yellow?logo=weightsandbiases)](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab)

---

## 목표

논문을 읽다가 수식이 나왔을 때 두려워하지 않고,
그 수식을 바로 NumPy / PyTorch 코드로 옮길 수 있는 상태를 만든다.

---

## 학습 내용 요약

### Week 1 — 선형대수 + NumPy

| 개념 | 수식 | 코드 |
|---|---|---|
| 내적 (dot product) | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ | `np.dot(a, b)` |
| 행렬 곱셈 | $C_{ij} = \sum_k A_{ik} B_{kj}$ | `A @ B` |
| einsum | `'ik,kj->ij'` | `np.einsum('ik,kj->ij', A, B)` |
| 브로드캐스팅 | $(m,n) + (n,)$ → $(m,n)$ | 자동 처리 |
| 코사인 유사도 | $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\Vert\mathbf{a}\Vert\Vert\mathbf{b}\Vert}$ | `X_norm @ X_norm.T` |

### Week 2 — 미분 · 역전파 · 옵티마이저

| 개념 | 수식 | 코드 |
|---|---|---|
| 연쇄법칙 | $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u}\cdot\frac{\partial u}{\partial x}$ | `_backward()` |
| 편미분 | $\frac{\partial L}{\partial w_i}$ (나머지 고정) | `param.grad` |
| 그래디언트 | $\nabla L = \left[\frac{\partial L}{\partial w_1}, \ldots\right]$ | `loss.backward()` |
| 경사하강법 | $w \leftarrow w - \eta \cdot \nabla L$ | `optimizer.step()` |
| Adam | $w \leftarrow w - \eta \cdot \hat{m}/\sqrt{\hat{v}}$ | `torch.optim.Adam` |
| AdamW | Adam + $w \mathrel{-}= \eta\lambda w$ (분리) | `torch.optim.AdamW` |

### Week 3 — 신경망 구현

| 단계 | 핵심 |
|---|---|
| NumPy MLP | 행렬 곱 + 역전파 직접 구현 |
| PyTorch MLP | autograd + AdamW + lr 스케줄러 |
| Lightning MLP | 학습 루프 추상화 + 콜백 + 체크포인트 |

---

## 구현 과제

### 01 — NumPy 선형대수 기초

[![Colab](https://img.shields.io/badge/Colab-열기-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/Wo0ond3r/portfolio-tsuruoka-lab/blob/main/phase0-math-python/01-numpy-linear-algebra/notebook/linear_algebra_basics.ipynb)
[![wandb](https://img.shields.io/badge/W%26B-실험-yellow?logo=weightsandbiases)](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab/runs/p6ok38yo?nw=nwuserswsw778)

**구현한 것:**
- 행렬 곱셈 3가지 방법 (`for loop` / `@` / `einsum`) + 속도 비교
- 브로드캐스팅 4가지 케이스 검증 (신경망 편향, Attention 마스크)
- 코사인 유사도 → 행렬 연산으로 N×N 한 번에 계산

**주요 결과:**

| 행렬 크기 | for loop | numpy @ | 속도 향상 |
|---|---|---|---|
| 16×16 | ~Xms | ~Yms | ~Zx |
| 64×64 | ~Xms | ~Yms | ~Zx |

> 📊 정확한 수치 → [W&B 실험](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab/runs/p6ok38yo?nw=nwuserswsw778)

```
01-numpy-linear-algebra/
├── notebook/   linear_algebra_basics.ipynb
├── src/        linalg_utils.py
├── docs/       선형대수_수식_정리.md
└── assets/     cosine_similarity_heatmap.png
```

---

### 02 — Micrograd: 자동미분 엔진

[![Colab](https://img.shields.io/badge/Colab-열기-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/YOUR_USERNAME/portfolio-tsuruoka-lab/blob/main/phase0-math-python/02-micrograd-from-scratch/notebook/micrograd_from_scratch.ipynb)
[![wandb](https://img.shields.io/badge/W%26B-실험-yellow?logo=weightsandbiases)](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab/runs/RUN_ID_02)

**구현한 것:**
- `Value` 클래스: 연산마다 `_backward()` 정의
- `backward()`: 위상 정렬 + 역순 순회 = PyTorch `.backward()` 원리
- 검증: micrograd == PyTorch (3케이스 일치)
- MLP로 XOR 학습: **100% 정확도**

**수식 → 코드 매핑:**

| 수식 | 코드 |
|---|---|
| $\partial c/\partial a = 1$ (덧셈) | `self.grad += out.grad` |
| $\partial c/\partial a = b$ (곱셈) | `self.grad += other.data * out.grad` |
| $\text{ReLU}' = [x>0]$ | `(out.data > 0) * out.grad` |
| 위상 정렬 후 역순 순회 | `for node in reversed(topo)` |
| $w \leftarrow w - \eta\,\partial L/\partial w$ | `p.data -= lr * p.grad` |

```
02-micrograd-from-scratch/
├── notebook/   micrograd_from_scratch.ipynb
├── src/        engine.py
├── docs/       역전파_수식_유도.md
└── assets/     xor_decision_boundary.png
```

---

### 03 — MLP MNIST: NumPy → PyTorch → Lightning

[![Colab](https://img.shields.io/badge/Colab-열기-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/YOUR_USERNAME/portfolio-tsuruoka-lab/blob/main/phase0-math-python/03-mlp-mnist/notebook/mlp_mnist.ipynb)
[![wandb](https://img.shields.io/badge/W%26B-실험-yellow?logo=weightsandbiases)](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab/runs/RUN_ID_03)

**구현한 것:**
- NumPy MLP: 행렬 곱셈 + 수동 역전파
- PyTorch MLP: AdamW + CosineAnnealingLR + grad clipping
- Lightning MLP: `training_step` / `configure_optimizers` / 콜백
- 하이퍼파라미터 스윕 4개 구성 실험

**세 구현 비교:**

| 구현 | Test Acc | 코드 줄 수 | 특징 |
|---|---|---|---|
| NumPy | ~95% | ~80줄 | 역전파 직접 구현 |
| PyTorch | ~98% | ~40줄 | AdamW + 스케줄러 |
| Lightning | ~98% | ~30줄 | 루프 추상화 |

> 📊 학습 곡선 + 스윕 결과 → [W&B 실험](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab/runs/RUN_ID_03)

**LLM 표준 학습 패턴 적용:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

```
03-mlp-mnist/
├── notebook/   mlp_mnist.ipynb
├── src/        model.py, train.py
├── docs/       MLP_원리_수식.md
└── assets/     training_curves.png  sweep_results.png
```

---

## Phase 0 → Phase 1 연결

| Phase 0에서 만든 것 | Phase 1에서 만나는 것 |
|---|---|
| `A @ B` 행렬 곱셈 | `nn.Conv2d` (슬라이딩 행렬 곱) |
| ReLU 역전파 `(x>0)*grad` | ResNet 50층 그래디언트 흐름 |
| `loss.backward()` 원리 | Batch Norm 역전파 |
| AdamW + CosineAnnealingLR | Phase 1~6 전체 학습 표준 |

**→ [Phase 1: Deep Learning 기초](../phase1-deep-learning/)**
