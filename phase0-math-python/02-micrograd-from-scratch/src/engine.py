"""
engine.py — Micrograd 자동미분 엔진
=====================================
Phase 0 - 구현 과제 2: micrograd-from-scratch

PyTorch의 Tensor (스칼라 버전)와 동일한 역할.
모든 연산이 계산 그래프를 자동으로 기록하고,
backward() 호출 시 연쇄법칙으로 그래디언트를 계산한다.

사용법:
    from src.engine import Value, backward

    x = Value(2.0, label='x')
    f = (x * 3 + 1) ** 2
    backward(f)
    print(x.grad)   # 42.0
"""

import math


class Value:
    """
    자동미분을 지원하는 스칼라 값.

    Attributes:
        data (float): 순전파 값
        grad (float): 역전파 그래디언트 (초기값 0)
        _op (str):    이 노드를 만든 연산 기호
        _prev (set):  이 노드의 입력 노드들
        label (str):  시각화용 이름 (선택)
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data      = float(data)
        self.grad      = 0.0
        self._op       = _op
        self._prev     = set(_children)
        self.label     = label
        self._backward = lambda: None   # 리프 노드 기본값

    # ── 덧셈 ──────────────────────────────────────────────────────────
    # 수식: c = a + b
    # 역전파: ∂c/∂a = 1, ∂c/∂b = 1  → 상류 grad 그대로 통과
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # ── 곱셈 ──────────────────────────────────────────────────────────
    # 수식: c = a * b
    # 역전파: ∂c/∂a = b, ∂c/∂b = a  → 서로 값을 바꿔서 곱함
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    # ── 거듭제곱 ──────────────────────────────────────────────────────
    # 수식: out = self ** n
    # 역전파: n * self^(n-1)  → xⁿ 미분 공식 그대로
    def __pow__(self, n):
        assert isinstance(n, (int, float)), "지수는 int 또는 float이어야 합니다"
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += (n * self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    # ── ReLU ──────────────────────────────────────────────────────────
    # 역전파: 양수면 그대로 통과(1), 음수면 차단(0)
    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    # ── tanh ──────────────────────────────────────────────────────────
    # 역전파: 1 - tanh²(x)
    def tanh(self):
        t   = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # ── exp ───────────────────────────────────────────────────────────
    # 역전파: exp(x) (자기 자신)
    def exp(self):
        e   = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # ── 편의 메서드 ───────────────────────────────────────────────────
    def __radd__(self, other): return self + other           # 3 + Value
    def __rmul__(self, other): return self * other           # 3 * Value
    def __neg__(self):         return self * -1              # -Value
    def __sub__(self, other):  return self + (-other)        # a - b
    def __rsub__(self, other): return Value(other) + (-self) # 3 - Value
    def __truediv__(self, other):  return self * other ** -1 # a / b
    def __rtruediv__(self, other): return Value(other) * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ── 역전파 함수 ────────────────────────────────────────────────────────

def backward(root: Value) -> list:
    """
    root 노드에서 시작해 역전파를 실행한다.
    = PyTorch의 loss.backward() 와 동일한 역할.

    Args:
        root: 역전파를 시작할 출력 노드 (Loss)

    Returns:
        topo: 위상 정렬된 노드 리스트 (시각화·디버깅용)

    동작 원리:
        1. 위상 정렬: DFS로 [리프 → 루트] 순서 결정
        2. root.grad = 1.0 (∂L/∂L = 1)
        3. reversed(topo): [루트 → 리프] 방향으로 _backward() 호출
    """
    # Step 1: 위상 정렬
    topo, visited = [], set()

    def build_topo(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:   # 자식(입력) 먼저
                build_topo(child)
            topo.append(node)          # 자신은 자식 처리 후

    build_topo(root)

    # Step 2: 출발 그래디언트
    root.grad = 1.0

    # Step 3: 역순으로 _backward() 호출 (연쇄법칙 적용)
    for node in reversed(topo):
        node._backward()

    return topo


def zero_grad(values: list):
    """모든 Value 노드의 grad를 0으로 초기화."""
    for v in values:
        v.grad = 0.0
