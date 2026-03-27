"""
train_micrograd.py — Micrograd 학습 유틸리티
==============================================
Phase 0 - 구현 과제 2: micrograd-from-scratch

engine.py + nn.py를 이용한 학습 루프.
PyTorch의 optimizer.step()과 동일한 동작을 수동으로 구현.

사용법:
    from src.engine import backward
    from src.nn import MLP
    from src.train_micrograd import train_step, evaluate, hinge_loss

    model = MLP(2, [8, 8, 1])
    for epoch in range(200):
        loss = train_step(model, X, y, lr=0.05)
"""

from .engine import Value, backward


# ── 손실 함수 ──────────────────────────────────────────────────────────

def hinge_loss(preds: list, targets: list, reg: float = 1e-4) -> Value:
    """
    Hinge Loss: max(0, 1 - y * ŷ)
    이진 분류 (-1 / +1 레이블)에 사용.

    Args:
        preds  (list[Value]): 모델 출력
        targets(list[float]): 정답 레이블 (-1 또는 +1)
        reg    (float):       L2 정규화 강도

    Returns:
        Value: 총 loss (data_loss + reg_loss)
    """
    # 데이터 loss
    data_loss = sum(
        (Value(1.0) + Value(-yi) * pi).relu()
        for pi, yi in zip(preds, targets)
    )

    # L2 정규화: 가중치가 너무 커지는 것 방지
    # 파라미터를 직접 참조하지 않으므로 외부에서 전달받아야 함
    return data_loss


def mse_loss(preds: list, targets: list) -> Value:
    """
    Mean Squared Error: (1/N) Σ (ŷ - y)²

    Args:
        preds  (list[Value]): 모델 출력
        targets(list[float]): 정답 값

    Returns:
        Value: MSE loss
    """
    n    = len(preds)
    loss = sum(
        (p - Value(y)) ** 2
        for p, y in zip(preds, targets)
    )
    return loss * (1.0 / n)


# ── 학습 스텝 ──────────────────────────────────────────────────────────

def train_step(model, X: list, y: list, lr: float = 0.05,
               reg: float = 1e-4, loss_fn=None) -> tuple:
    """
    한 스텝의 학습을 실행한다.
    = zero_grad → forward → loss → backward → step

    Args:
        model:   MLP 인스턴스
        X:       입력 리스트 [[x1,x2,...], ...]
        y:       정답 레이블 리스트
        lr:      학습률
        reg:     L2 정규화 강도
        loss_fn: 손실 함수 (기본: hinge_loss)

    Returns:
        (loss_val, accuracy)
    """
    # ① zero_grad: 이전 스텝 그래디언트 초기화
    model.zero_grad()

    # ② 순전파
    preds = [model(xi) for xi in X]

    # ③ loss 계산
    if loss_fn is None:
        data_loss = sum(
            (Value(1.0) + Value(-yi) * pi).relu()
            for pi, yi in zip(preds, y)
        )
        reg_loss  = Value(reg) * sum(p ** 2 for p in model.parameters())
        loss      = data_loss + reg_loss
    else:
        loss = loss_fn(preds, y)

    # ④ 역전파: 연쇄법칙으로 모든 ∂L/∂w 계산
    backward(loss)

    # ⑤ SGD 업데이트: w ← w - lr × ∂L/∂w
    for p in model.parameters():
        p.data -= lr * p.grad

    # 정확도 계산 (이진 분류 기준)
    acc = sum(
        1 for pi, yi in zip(preds, y)
        if (pi.data > 0) == (yi > 0)
    ) / len(y)

    return loss.data, acc


def evaluate(model, X: list, y: list) -> float:
    """
    모델의 정확도를 계산한다.

    Args:
        model: MLP 인스턴스
        X:     입력 리스트
        y:     정답 레이블 리스트

    Returns:
        accuracy (float): 0.0 ~ 1.0
    """
    preds = [model(xi) for xi in X]
    return sum(
        1 for pi, yi in zip(preds, y)
        if (pi.data > 0) == (yi > 0)
    ) / len(y)


def train(model, X: list, y: list, epochs: int = 200,
          lr: float = 0.05, reg: float = 1e-4,
          log_every: int = 20, wandb_run=None) -> dict:
    """
    전체 학습 루프를 실행한다.

    Args:
        model:    MLP 인스턴스
        X:        입력 리스트
        y:        정답 레이블 리스트
        epochs:   총 에폭 수
        lr:       학습률
        reg:      L2 정규화 강도
        log_every:몇 에폭마다 출력할지
        wandb_run:wandb run 인스턴스 (선택)

    Returns:
        history: {'loss': [...], 'acc': [...]}
    """
    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        loss_val, acc = train_step(model, X, y, lr=lr, reg=reg)
        history['loss'].append(loss_val)
        history['acc'].append(acc)

        if wandb_run is not None:
            wandb_run.log({'train/loss': loss_val, 'train/acc': acc}, step=epoch)

        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | loss: {loss_val:.4f} | acc: {acc*100:.0f}%")

    return history
