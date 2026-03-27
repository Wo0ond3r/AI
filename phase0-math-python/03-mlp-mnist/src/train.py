"""
train.py — MLP MNIST 학습 유틸리티
=====================================
Phase 0 - 구현 과제 3: mlp-mnist

NumPy MLP와 PyTorch MLP의 학습 루프를 제공한다.
Lightning MLP는 Trainer.fit()으로 처리되므로 여기서 다루지 않는다.

사용법:
    from src.model import NumpyMLP, PyTorchMLP
    from src.train import train_numpy, train_pytorch, get_dataloaders

    # NumPy
    model   = NumpyMLP(lr=0.05)
    history = train_numpy(model, X_train, y_train, X_test, y_test, epochs=20)

    # PyTorch
    train_loader, test_loader = get_dataloaders(batch_size=256)
    model   = PyTorchMLP().to('cuda')
    history = train_pytorch(model, train_loader, test_loader, epochs=10)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import NumpyMLP, PyTorchMLP


# ── 데이터 로더 ────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256,
                    data_dir: str = '/tmp/mnist') -> tuple:
    """
    MNIST 데이터 로더를 반환한다.

    Args:
        batch_size (int): 미니배치 크기
        data_dir   (str): 데이터 저장 경로

    Returns:
        (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 평균/표준편차
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False, num_workers=2)
    return train_loader, test_loader


def get_numpy_data(n_train: int = 10000,
                   data_dir: str = '/tmp/mnist') -> tuple:
    """
    MNIST를 NumPy 배열로 반환한다.

    Returns:
        (X_train, y_train, X_test, y_test) — 픽셀값 0~1 정규화
    """
    transform = transforms.ToTensor()
    train_ds  = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds   = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    X_tr = train_ds.data[:n_train].numpy().reshape(n_train, -1) / 255.0
    y_tr = train_ds.targets[:n_train].numpy()
    X_te = test_ds.data.numpy().reshape(-1, 784) / 255.0
    y_te = test_ds.targets.numpy()
    return X_tr, y_tr, X_te, y_te


# ── NumPy 학습 루프 ────────────────────────────────────────────────────

def train_numpy(model: NumpyMLP,
                X_train: np.ndarray, y_train: np.ndarray,
                X_test:  np.ndarray, y_test:  np.ndarray,
                epochs: int = 20, batch_size: int = 256,
                log_every: int = 5, wandb_run=None) -> dict:
    """
    NumPy MLP 학습 루프.

    Args:
        model:      NumpyMLP 인스턴스
        X_train:    (N, 784) 학습 데이터
        y_train:    (N,) 학습 레이블
        X_test:     (M, 784) 테스트 데이터
        y_test:     (M,) 테스트 레이블
        epochs:     총 에폭 수
        batch_size: 미니배치 크기
        log_every:  몇 에폭마다 출력할지
        wandb_run:  wandb run 인스턴스 (선택)

    Returns:
        history: {'train_loss': [...], 'test_acc': [...]}
    """
    n_train = len(X_train)
    history = {'train_loss': [], 'test_acc': []}
    rng     = np.random.default_rng(42)

    for epoch in range(epochs):
        # 셔플
        idx    = rng.permutation(n_train)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        # 미니배치 루프
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, n_train, batch_size):
            xb = X_shuf[i:i + batch_size]
            yb = y_shuf[i:i + batch_size]
            model.forward(xb)
            epoch_loss += model.loss(yb)
            model.backward(yb)
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        test_acc = model.accuracy(X_test, y_test)

        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)

        if wandb_run is not None:
            wandb_run.log(
                {'numpy/loss': avg_loss, 'numpy/test_acc': test_acc},
                step=epoch
            )

        if epoch % log_every == 0:
            print(f"[NumPy] Epoch {epoch:3d} | loss: {avg_loss:.4f} | test acc: {test_acc*100:.1f}%")

    return history


# ── PyTorch 학습 루프 ──────────────────────────────────────────────────

def train_pytorch(model: PyTorchMLP,
                  train_loader: DataLoader,
                  test_loader:  DataLoader,
                  epochs:       int   = 10,
                  lr:           float = 1e-3,
                  weight_decay: float = 0.01,
                  device:       str   = 'cpu',
                  log_every:    int   = 1,
                  wandb_run=None) -> dict:
    """
    PyTorch MLP 학습 루프 (LLM 표준 패턴 적용).

    학습 루프 수식 ↔ 코드 대응:
        ∂L/∂w = 0 초기화    →  optimizer.zero_grad()
        ŷ = model(x)       →  pred = model(x)
        L = CE(ŷ, y)       →  loss = criterion(pred, y)
        ∀i: ∂L/∂wᵢ 계산   →  loss.backward()
        ‖∇L‖ > 1 스케일다운 →  clip_grad_norm_(1.0)
        wᵢ ← AdamW 수식    →  optimizer.step()
        η_t = cosine(t)    →  scheduler.step()

    Args:
        model:        PyTorchMLP 인스턴스
        train_loader: 학습 DataLoader
        test_loader:  테스트 DataLoader
        epochs:       총 에폭 수
        lr:           학습률
        weight_decay: AdamW weight decay
        device:       'cpu' 또는 'cuda'
        log_every:    몇 에폭마다 출력할지
        wandb_run:    wandb run 인스턴스 (선택)

    Returns:
        history: {'train_loss': [...], 'test_acc': [...], 'lr': [...]}
    """
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # AdamW + CosineAnnealingLR — LLM 표준 패턴
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    history = {'train_loss': [], 'test_acc': [], 'lr': []}

    for epoch in range(epochs):
        # ── 학습 ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()                             # ∂L/∂w = 0
            pred = model(x)                                   # 순전파
            loss = criterion(pred, y)                         # Loss 계산
            loss.backward()                                   # 역전파
            torch.nn.utils.clip_grad_norm_(                   # grad clipping
                model.parameters(), max_norm=1.0
            )
            optimizer.step()                                  # AdamW 업데이트
            train_loss += loss.item()

        scheduler.step()                                      # lr 코사인 감소

        # ── 평가 ──────────────────────────────────────────────────────
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y     = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()

        avg_loss = train_loss / len(train_loader)
        test_acc = correct / len(test_loader.dataset)
        cur_lr   = scheduler.get_last_lr()[0]

        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(cur_lr)

        if wandb_run is not None:
            wandb_run.log(
                {'pytorch/loss': avg_loss,
                 'pytorch/test_acc': test_acc,
                 'pytorch/lr': cur_lr},
                step=epoch
            )

        if epoch % log_every == 0:
            print(f"[PyTorch] Epoch {epoch+1:3d} | loss: {avg_loss:.4f} | "
                  f"acc: {test_acc*100:.2f}% | lr: {cur_lr:.5f}")

    return history


# ── 하이퍼파라미터 그리드 서치 ────────────────────────────────────────

def grid_search(train_loader: DataLoader, test_loader: DataLoader,
                configs: list, epochs: int = 5,
                device: str = 'cpu', wandb_run=None) -> list:
    """
    여러 하이퍼파라미터 조합을 실험한다.

    Args:
        train_loader: 학습 DataLoader
        test_loader:  테스트 DataLoader
        configs: [{'lr': ..., 'wd': ..., 'dropout': ...}, ...]
        epochs:  각 설정의 학습 에폭 수
        device:  'cpu' 또는 'cuda'
        wandb_run: wandb run (선택)

    Returns:
        results: [{'lr': ..., 'wd': ..., 'dropout': ..., 'acc': ...}, ...]
    """
    results = []

    for cfg in configs:
        print(f"\n실험: lr={cfg['lr']}, wd={cfg['wd']}, dropout={cfg['dropout']}")
        model   = PyTorchMLP(dropout=cfg['dropout']).to(device)
        history = train_pytorch(
            model, train_loader, test_loader,
            epochs=epochs, lr=cfg['lr'],
            weight_decay=cfg['wd'], device=device,
            log_every=epochs,   # 마지막 에폭만 출력
        )
        acc = history['test_acc'][-1]
        results.append({**cfg, 'acc': acc})

        if wandb_run is not None:
            wandb_run.log({
                'sweep/acc':     acc,
                'sweep/lr':      cfg['lr'],
                'sweep/wd':      cfg['wd'],
                'sweep/dropout': cfg['dropout'],
            })
        print(f"  → acc: {acc*100:.2f}%")

    best = max(results, key=lambda x: x['acc'])
    print(f"\n최적 설정: {best}")
    return results
