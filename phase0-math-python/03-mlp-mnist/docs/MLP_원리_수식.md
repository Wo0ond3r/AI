# MLP 원리 & 학습 수식 정리 — Phase 0 Week 3

> **목적:** 신경망 한 층이 수식으로 어떻게 생겼는지, 학습 루프의 각 줄이
> 어떤 수식에 해당하는지 언제든 꺼내볼 수 있는 레퍼런스.
> NumPy → PyTorch → Lightning 세 구현을 수식으로 연결해서 기록한다.

---

## 1. 신경망 한 층의 수식

### 수식

$$\mathbf{z} = \mathbf{x} W + \mathbf{b}, \quad \mathbf{a} = \sigma(\mathbf{z})$$

- $\mathbf{x}$: 입력 벡터 (또는 배치 행렬)
- $W$: 가중치 행렬 (학습 대상)
- $\mathbf{b}$: 편향 벡터 (브로드캐스팅으로 더해짐)
- $\sigma$: 활성화 함수 (ReLU, GELU 등)

### Shape 추적 (배치 처리)

```
입력 x:  (N, D_in)    ← N개 샘플, D_in 차원
가중치 W: (D_in, D_out)
편향 b:   (D_out,)

z = x @ W + b
  = (N, D_in) @ (D_in, D_out) + (D_out,)
  = (N, D_out)               + (D_out,)   ← 브로드캐스팅!
  = (N, D_out)
```

### 코드 대응

```python
# NumPy
Z = X @ W + b                    # 과제 1 내용 그대로

# PyTorch
layer = nn.Linear(D_in, D_out)   # W, b를 내부에서 관리
Z = layer(X)                     # 동일한 연산

# Lightning
self.net = nn.Sequential(
    nn.Linear(784, 256),         # 한 층 = 수식 한 줄
    nn.ReLU(),
    ...
)
```

---

## 2. MLP 전체 순전파

### 2층 MLP 수식 (784 → 256 → 128 → 10)

$$\mathbf{z}_1 = \mathbf{x} W_1 + \mathbf{b}_1, \quad \mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$$

$$\mathbf{z}_2 = \mathbf{a}_1 W_2 + \mathbf{b}_2, \quad \mathbf{a}_2 = \text{ReLU}(\mathbf{z}_2)$$

$$\mathbf{z}_3 = \mathbf{a}_2 W_3 + \mathbf{b}_3, \quad \hat{\mathbf{y}} = \text{Softmax}(\mathbf{z}_3)$$

### Shape 흐름

```
입력:   (N, 784)
→ W1:   (N, 256)   ReLU
→ W2:   (N, 128)   ReLU
→ W3:   (N, 10)    Softmax
출력:   (N, 10)    ← 10 클래스 확률
```

---

## 3. 손실 함수

### CrossEntropy Loss

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{y}_{i, c_i}$$

- $\hat{y}_{i, c_i}$: i번째 샘플의 정답 클래스 $c_i$ 에 대한 예측 확률
- 확률이 높을수록(잘 맞출수록) loss가 낮아짐

### 직관

```
정답 클래스 확률 0.9  →  -log(0.9) ≈ 0.105  ← loss 작음 (잘 예측)
정답 클래스 확률 0.1  →  -log(0.1) ≈ 2.303  ← loss 큼  (잘못 예측)
```

### 코드

```python
# NumPy
loss = -np.log(out[np.arange(N), y] + 1e-8).mean()

# PyTorch (내부적으로 softmax 포함)
loss = nn.CrossEntropyLoss()(logits, y)
```

---

## 4. 역전파 수식 (MLP)

### Softmax + CrossEntropy 합산 역전파

$$\frac{\partial L}{\partial \mathbf{z}_3} = \frac{1}{N}(\hat{\mathbf{y}} - \mathbf{y}_{\text{one-hot}})$$

직관: 정답 클래스의 확률이 높을수록 그래디언트가 작아진다.

### 각 층의 역전파

$$\frac{\partial L}{\partial W_k} = \mathbf{a}_{k-1}^\top \cdot \frac{\partial L}{\partial \mathbf{z}_k}$$

$$\frac{\partial L}{\partial \mathbf{b}_k} = \sum_{\text{배치}} \frac{\partial L}{\partial \mathbf{z}_k}$$

$$\frac{\partial L}{\partial \mathbf{a}_{k-1}} = \frac{\partial L}{\partial \mathbf{z}_k} \cdot W_k^\top$$

### ReLU 역전파

$$\frac{\partial L}{\partial \mathbf{z}_k} = \frac{\partial L}{\partial \mathbf{a}_k} \odot \mathbf{1}[\mathbf{z}_k > 0]$$

$\odot$: 원소별 곱. 음수였던 뉴런은 그래디언트 차단.

### NumPy 코드 대응

```python
# Softmax+CE 역전파
dZ3 = out.copy()
dZ3[np.arange(N), y] -= 1
dZ3 /= N                          # ∂L/∂z₃ = (ŷ - y) / N

# 3층 → 파라미터 그래디언트
dW3 = A2.T @ dZ3                  # ∂L/∂W₃ = A₂ᵀ @ ∂L/∂z₃
db3 = dZ3.sum(axis=0)             # ∂L/∂b₃ (배치 합산)
dA2 = dZ3 @ W3.T                  # 아래 층으로 전달

# ReLU 역전파
dZ2 = dA2 * (Z2 > 0)              # ∂L/∂z₂ = ∂L/∂a₂ ⊙ 1[z₂>0]
```

---

## 5. He 초기화

### 수식

$$W \sim \mathcal{N}\!\left(0,\, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

### 왜 이 값인가

ReLU는 입력의 절반(음수)을 0으로 만들어 분산을 절반으로 줄인다.
이를 보상하기 위해 $\sqrt{2/n_{\text{in}}}$ 으로 초기화.
→ 레이어를 거쳐도 분산이 폭발하거나 소실되지 않는다.

```python
# NumPy
W1 = np.random.randn(784, 256) * np.sqrt(2/784)

# PyTorch nn.Linear는 내부적으로 Kaiming(He) 초기화 사용
```

---

## 6. 옵티마이저 수식

### SGD

$$w \leftarrow w - \eta \cdot \nabla L$$

단순. 학습률 튜닝에 민감.

### SGD + 모멘텀

$$v_t = \beta v_{t-1} + (1-\beta)\nabla L$$
$$w \leftarrow w - \eta v_t$$

관성으로 진동 감소, 수렴 속도 향상. $\beta = 0.9$ 가 기본값.

### Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1차 모멘텀)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2차 모멘텀)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(편향 보정)}$$
$$w \leftarrow w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

파라미터별 적응형 학습률. $\beta_1=0.9,\, \beta_2=0.999,\, \epsilon=10^{-8}$이 기본값.

### AdamW (Adam과의 차이)

$$w \leftarrow w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta\lambda w$$

마지막 항 $\eta\lambda w$이 **별도로** 빠져나온 것이 AdamW.
weight decay가 적응형 lr에 의해 왜곡되지 않는다 → 더 나은 일반화.

```python
# 코드 한 줄 차이
torch.optim.Adam(model.parameters(),   lr=1e-3, weight_decay=0.01)  # 섞임
torch.optim.AdamW(model.parameters(),  lr=1e-3, weight_decay=0.01)  # 분리 ✓
```

---

## 7. 학습률 스케줄러

### CosineAnnealingLR

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

- 초반: 학습률 높음 → 빠른 수렴
- 후반: 학습률 낮음 → 정밀 수렴
- LLM 학습의 표준 패턴

```python
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
# 매 에폭 끝에 scheduler.step() 호출
```

---

## 8. 그래디언트 클리핑

### 수식

$$\text{if } \|\nabla\| > \text{max\_norm}: \quad \nabla \leftarrow \nabla \cdot \frac{\text{max\_norm}}{\|\nabla\|}$$

전체 그래디언트의 크기가 임계값을 넘으면 비율에 맞게 축소.
개별 그래디언트를 자르는 것이 아니라 **전체 방향은 유지**하면서 크기만 줄임.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# loss.backward() 다음, optimizer.step() 이전에 호출
```

Transformer 학습 초기에 그래디언트 폭발을 막는 표준 패턴.

---

## 9. 학습 루프 수식 ↔ 코드 완전 매핑

```
수식                            PyTorch 코드
──────────────────────────────────────────────────────
∂L/∂w_i = 0 초기화          →  optimizer.zero_grad()
ŷ = f(x; W)                 →  pred = model(x)
L = loss(ŷ, y)              →  loss = criterion(pred, y)
∀i: ∂L/∂w_i 계산            →  loss.backward()
‖∇L‖ > 1이면 정규화          →  clip_grad_norm_(..., 1.0)
w_i ← w_i - η·update(∂L/∂w_i) → optimizer.step()
η_t = cosine(t)              →  scheduler.step()
```

---

## 10. Batch Normalization (보너스)

### 수식

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma\hat{x} + \beta$$

- $\mu_B, \sigma_B^2$: 미니배치의 평균과 분산
- $\gamma, \beta$: 학습 가능한 스케일/시프트 파라미터
- 각 레이어 출력을 정규화해서 학습 안정화

```python
nn.Sequential(
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),   # ← 정규화
    nn.ReLU(),
)
```

역전파: $\gamma, \beta$도 학습 파라미터이므로 `.grad`가 자동 계산됨.
단, weight decay 적용 대상에서 **제외**하는 것이 표준:

```python
no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm']
# AdamW 파라미터 그룹 분리 패턴 (Phase 3 BERT 코드에서 다시 등장)
```

---

## 11. NumPy → PyTorch → Lightning 대응표

| 개념 | NumPy | PyTorch | Lightning |
|---|---|---|---|
| 순전파 | `Z = X @ W + b` | `nn.Linear` | `self.net(x)` |
| 활성화 | `np.maximum(0,z)` | `nn.ReLU()` | (동일) |
| 역전파 | 수동 `dW = A.T @ dZ` | `loss.backward()` | `training_step` 리턴 |
| 파라미터 업데이트 | `W -= lr * dW` | `optimizer.step()` | `configure_optimizers` |
| 그래드 초기화 | `dW = 0` | `optimizer.zero_grad()` | (자동) |
| 평가 모드 | — | `model.eval()` | `validation_step` |
| 체크포인트 | — | 직접 구현 | `ModelCheckpoint` 콜백 |
| 조기 종료 | — | 직접 구현 | `EarlyStopping` 콜백 |

---

## 12. AI로의 연결

| 이번 과제에서 배운 것 | Phase 2+ 에서 만나는 곳 |
|---|---|
| `nn.Linear` (행렬 곱 + 편향) | Transformer Q, K, V 선형 변환 |
| Softmax + CrossEntropy | 언어 모델 다음 토큰 예측 loss |
| He 초기화 | GPT 가중치 초기화 |
| AdamW + CosineAnnealingLR | BERT, GPT-2 ~ LLaMA 전부 |
| `clip_grad_norm_` | Transformer 학습 폭발 방지 |
| BatchNorm → LayerNorm | Transformer는 LayerNorm 사용 |
| `no_decay` 파라미터 그룹 | BERT fine-tuning 표준 코드 |

---

## 메모: 이해 안 됐던 부분과 해결

| 날짜 | 수식/개념 | 어디서 막혔나 | 해결 방법 |
|---|---|---|---|
| YYYY-MM-DD | | | |
