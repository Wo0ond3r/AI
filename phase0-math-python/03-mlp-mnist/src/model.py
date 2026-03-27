"""
model.py — MLP MNIST 모델 정의
================================
Phase 0 - 구현 과제 3: mlp-mnist

NumPy MLP, PyTorch MLP, Lightning MLP 세 버전의 모델을 정의한다.
세 구현이 같은 구조를 갖도록 통일된 하이퍼파라미터를 사용한다.

사용법:
    from src.model import NumpyMLP, PyTorchMLP, LightningMLP

    # NumPy
    model = NumpyMLP(lr=0.05)

    # PyTorch
    model = PyTorchMLP(dropout=0.3).to(device)

    # Lightning
    model = LightningMLP(lr=1e-3, weight_decay=0.01, dropout=0.3)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ── 공통 하이퍼파라미터 ────────────────────────────────────────────────

INPUT_DIM  = 784    # 28×28 픽셀
HIDDEN1    = 256
HIDDEN2    = 128
OUTPUT_DIM = 10     # 0~9 클래스


# ──────────────────────────────────────────────────────────────────────
# 1. NumPy MLP
# ──────────────────────────────────────────────────────────────────────

class NumpyMLP:
    """
    NumPy로 구현한 2층 MLP. 역전파를 직접 구현한다.

    구조: 784 → 256 → 128 → 10

    수식 연결:
        순전파: Z = X @ W + b   (과제 1 행렬 곱셈 + 브로드캐스팅)
        역전파: dW = X.T @ dZ  (과제 2 연쇄법칙 행렬 버전)
    """

    def __init__(self, lr: float = 0.05):
        self.lr = lr
        rng     = np.random.default_rng(42)

        # He 초기화: ReLU에 최적화 (분산 = 2 / fan_in)
        self.W1 = rng.standard_normal((INPUT_DIM, HIDDEN1)) * math.sqrt(2 / INPUT_DIM)
        self.b1 = np.zeros(HIDDEN1)
        self.W2 = rng.standard_normal((HIDDEN1,  HIDDEN2)) * math.sqrt(2 / HIDDEN1)
        self.b2 = np.zeros(HIDDEN2)
        self.W3 = rng.standard_normal((HIDDEN2, OUTPUT_DIM)) * math.sqrt(2 / HIDDEN2)
        self.b3 = np.zeros(OUTPUT_DIM)

        # 순전파 캐시 (역전파에서 사용)
        self.X = self.Z1 = self.A1 = None
        self.Z2 = self.A2 = self.Z3 = self.out = None

    # ── 활성화 함수 ────────────────────────────────────────────────────

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        # 수치 안정성: max 빼기 (브로드캐스팅!)
        z     = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # ── 순전파 ─────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        순전파: X → Z1 → A1 → Z2 → A2 → Z3 → softmax

        Args:
            X: (N, 784) 입력 배치

        Returns:
            out: (N, 10) 클래스별 확률
        """
        self.X   = X
        self.Z1  = X  @ self.W1 + self.b1   # (N,784)@(784,256) = (N,256)
        self.A1  = self.relu(self.Z1)
        self.Z2  = self.A1 @ self.W2 + self.b2  # (N,256)@(256,128) = (N,128)
        self.A2  = self.relu(self.Z2)
        self.Z3  = self.A2 @ self.W3 + self.b3  # (N,128)@(128,10)  = (N,10)
        self.out = self.softmax(self.Z3)
        return self.out

    # ── 손실 함수 ──────────────────────────────────────────────────────

    def loss(self, y: np.ndarray) -> float:
        """CrossEntropy: -log(정답 클래스 확률)의 평균."""
        N = len(y)
        return -np.log(self.out[np.arange(N), y] + 1e-8).mean()

    # ── 역전파 + 파라미터 업데이트 ────────────────────────────────────

    def backward(self, y: np.ndarray):
        """
        역전파 수동 구현 (연쇄법칙).

        Args:
            y: (N,) 정답 레이블
        """
        N = len(y)

        # Softmax + CrossEntropy 합산 역전파: ŷ - one_hot(y)
        dZ3 = self.out.copy()
        dZ3[np.arange(N), y] -= 1
        dZ3 /= N

        # 3층 역전파
        dW3 = self.A2.T @ dZ3            # ∂L/∂W3 = A2ᵀ @ dZ3
        db3 = dZ3.sum(axis=0)            # ∂L/∂b3 (브로드캐스팅 역전파)
        dA2 = dZ3 @ self.W3.T

        # ReLU 역전파: 양수만 통과
        dZ2 = dA2 * (self.Z2 > 0)
        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T

        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        # SGD 업데이트: w ← w - lr × ∂L/∂w
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    # ── 평가 ───────────────────────────────────────────────────────────

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.forward(X).argmax(axis=1)
        return (preds == y).mean()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)


# ──────────────────────────────────────────────────────────────────────
# 2. PyTorch MLP
# ──────────────────────────────────────────────────────────────────────

class PyTorchMLP(nn.Module):
    """
    PyTorch nn.Module 버전 MLP.
    NumPy 버전과 동일한 구조, autograd로 역전파 자동화.

    구조: 784 → 256 (BN+ReLU+DO) → 128 (BN+ReLU+DO) → 10

    nn.Linear = 행렬 곱셈(W) + 브로드캐스팅(b)를 하나로 묶은 것.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(HIDDEN2, OUTPUT_DIM),
        )
        self._init_weights()

    def _init_weights(self):
        """He 초기화 (Linear 레이어)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))   # flatten → MLP

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────
# 3. Lightning MLP
# ──────────────────────────────────────────────────────────────────────

class LightningMLP(pl.LightningModule):
    """
    PyTorch Lightning 버전 MLP.
    training_step / validation_step / configure_optimizers 만 정의.

    하이퍼파라미터:
        lr           (float): 학습률 (기본: 1e-3)
        weight_decay (float): AdamW weight decay (기본: 0.01)
        dropout      (float): 드롭아웃 비율 (기본: 0.3)
        t_max        (int):   CosineAnnealingLR 주기 (기본: 10)
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.01,
                 dropout: float = 0.3, t_max: int = 10):
        super().__init__()
        self.save_hyperparameters()   # wandb 자동 연동

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(HIDDEN2, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def _shared_step(self, batch):
        x, y  = batch
        logits = self(x)
        loss   = F.cross_entropy(logits, y)
        acc    = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc',  acc,  prog_bar=True)

    def configure_optimizers(self):
        """AdamW + CosineAnnealingLR — LLM 표준 패턴."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.t_max
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
