"""
nn.py — Micrograd 기반 신경망 레이어
======================================
Phase 0 - 구현 과제 2: micrograd-from-scratch

engine.py의 Value 위에 Neuron, Layer, MLP를 구현한다.
모든 가중치가 Value이므로 backward()가 자동으로 작동한다.

사용법:
    from src.engine import backward
    from src.nn import MLP

    model = MLP(2, [8, 8, 1])
    out   = model([0.5, 0.3])
    backward(out)
"""

import random
from .engine import Value


class Neuron:
    """
    단일 뉴런: y = activation(w·x + b)

    Args:
        n_inputs   (int): 입력 차원
        activation (str): 'relu' | 'tanh' | 'linear'
    """

    def __init__(self, n_inputs: int, activation: str = 'relu'):
        self.w          = [Value(random.uniform(-1.0, 1.0)) for _ in range(n_inputs)]
        self.b          = Value(0.0)
        self.activation = activation

    def __call__(self, x: list) -> Value:
        # w·x + b  (내적 = 행렬 곱셈의 스칼라 버전)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.activation == 'relu':   return act.relu()
        if self.activation == 'tanh':   return act.tanh()
        return act   # linear

    def parameters(self) -> list:
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return f"Neuron(n_in={len(self.w)}, act={self.activation})"


class Layer:
    """
    뉴런 n_outputs 개를 병렬로 실행하는 층.

    Args:
        n_inputs  (int): 입력 차원
        n_outputs (int): 출력 차원 (뉴런 수)
        activation(str): 각 뉴런의 활성화 함수
    """

    def __init__(self, n_inputs: int, n_outputs: int, activation: str = 'relu'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]

    def __call__(self, x: list) -> list:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list:
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


class MLP:
    """
    다층 퍼셉트론 (Multi-Layer Perceptron).

    Args:
        n_inputs   (int):  입력 차원
        layer_sizes(list): 각 층의 출력 크기
                           마지막 층은 자동으로 'linear' 활성화

    예시:
        MLP(2, [8, 8, 1])   # 2 → 8 → 8 → 1
        MLP(784, [256, 128, 10])
    """

    def __init__(self, n_inputs: int, layer_sizes: list):
        sizes      = [n_inputs] + layer_sizes
        self.layers = [
            Layer(
                sizes[i], sizes[i + 1],
                activation='relu' if i < len(layer_sizes) - 1 else 'linear'
            )
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x: list):
        for layer in self.layers:
            x = layer(x)
        # 출력이 1개이면 Value, 아니면 list 반환
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """모든 파라미터의 grad를 0으로 초기화. = optimizer.zero_grad()"""
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return f"MLP([{', '.join(str(l) for l in self.layers)}])"
