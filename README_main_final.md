# 🎓 Portfolio — Tsuruoka Lab (Tokyo University)

> **목표:** 도쿄대학 Tsuruoka 연구실 (NLP · 강화학습 · 게임 AI) 진학을 위한 연구 포트폴리오
>
> **방향:** 기초 딥러닝부터 LLM × Game AI까지, 논문 구현 중심으로 구성

[![wandb](https://img.shields.io/badge/Experiments-W%26B-yellow?logo=weightsandbiases)](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-purple?logo=lightning)](https://lightning.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🗺️ 전체 커리큘럼 로드맵

```
Phase 0  수학 & Python 기초          ██████████  ✅ 완료
Phase 1  Deep Learning 기초          ░░░░░░░░░░  진행 예정
Phase 2  NLP & Transformers          ░░░░░░░░░░  진행 예정
Phase 3  Large Language Models       ░░░░░░░░░░  진행 예정
Phase 4  RL & Game AI                ░░░░░░░░░░  진행 예정
Phase 5  LLM × Games  ★             ░░░░░░░░░░  진행 예정
Phase 6  SOTA & 독자적 연구           ░░░░░░░░░░  진행 예정
```

---

## 📁 프로젝트 구조

```
portfolio-tsuruoka-lab/
│
├── phase0-math-python/                  ✅ 완료
│   ├── 01-numpy-linear-algebra/         ✅ 행렬 곱셈 · 브로드캐스팅 · 코사인 유사도
│   ├── 02-micrograd-from-scratch/       ✅ 자동미분 엔진 · XOR 학습
│   └── 03-mlp-mnist/                    ✅ NumPy → PyTorch → Lightning · 하이퍼파라미터 스윕
│
├── phase1-deep-learning/                ⬜ 예정
│   ├── 01-resnet-from-scratch/
│   └── 02-batch-norm/
│
├── phase2-nlp-transformers/             ⬜ 예정
│   ├── 01-transformer-from-scratch/
│   └── 02-bert-pretrain/
│
├── phase3-llm/                          ⬜ 예정
│   ├── 01-rlhf-minimal/
│   └── 02-lora-finetune/
│
├── phase4-rl-game-ai/                   ⬜ 예정
│   ├── 01-dqn-atari/
│   └── 02-alphazero-connect4/
│
├── phase5-llm-x-games/                  ★ 핵심 목표
│   ├── 01-react-agent/                  ⬜ 예정
│   ├── 02-tot-game-planning/            ⬜ 예정
│   ├── 03-mahjong-value-net/            ⬜ 예정  (Tsuruoka Lab CoG 2024)
│   └── 04-llm-reasoning-tuning/         ⬜ 예정  (Tsuruoka Lab arXiv 2024)
│
└── phase6-research/                     ★ 독자적 연구
    └── 01-consistency-offline-rl/       ⬜ 예정  (Tsuruoka Lab TMLR)
```

---

## ✅ Phase 0 — 수학 & Python 기초 (완료)

### 구현 과제 요약

| # | 주제 | 핵심 구현 | wandb | 노트북 |
|---|---|---|---|---|
| 01 | NumPy 선형대수 | 행렬 곱셈 (3가지) · 브로드캐스팅 · 코사인 유사도 | [링크](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab/runs/RUN_ID_01) | [Colab](https://colab.research.google.com/github/YOUR_USERNAME/portfolio-tsuruoka-lab/blob/main/phase0-math-python/01-numpy-linear-algebra/notebook/linear_algebra_basics.ipynb) |
| 02 | Micrograd | 자동미분 엔진 · XOR 100% 정확도 | [링크](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab/runs/RUN_ID_02) | [Colab](https://colab.research.google.com/github/YOUR_USERNAME/portfolio-tsuruoka-lab/blob/main/phase0-math-python/02-micrograd-from-scratch/notebook/micrograd_from_scratch.ipynb) |
| 03 | MLP MNIST | NumPy → PyTorch → Lightning · AdamW · 스윕 | [링크](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab/runs/RUN_ID_03) | [Colab](https://colab.research.google.com/github/YOUR_USERNAME/portfolio-tsuruoka-lab/blob/main/phase0-math-python/03-mlp-mnist/notebook/mlp_mnist.ipynb) |

### Phase 0에서 배운 것 → AI 연결

| 개념 | 구현한 것 | Phase 5에서 만나는 곳 |
|---|---|---|
| 행렬 곱셈 `A @ B` | `matmul_loop / @ / einsum` | Transformer `Q @ K.T` |
| 브로드캐스팅 | `(B,H,T,T) + (T,T)` | Attention mask |
| 연쇄법칙 | `Value._backward()` | `loss.backward()` |
| 그래디언트 | `∂L/∂W = Xᵀ @ dZ` | 역전파 전체 |
| AdamW | `weight_decay 직접 적용` | GPT, LLaMA 표준 옵티마이저 |

---

## 🎯 목표 연구실

**[Tsuruoka Laboratory](https://www.logos.t.u-tokyo.ac.jp/)** — 도쿄대학 대학원 정보이공학계연구과

> *"Natural Language Processing · Reinforcement Learning · Artificial Intelligence for Games"*

### 구현 대상 논문 (Phase 5~6)

| 논문 | 발표 | 구현 단계 |
|---|---|---|
| MJ-DLVAT: Deep Learning Value Assessment for Mahjong | CoG 2024 | Phase 5 |
| Improving Arithmetic Reasoning via Relation Tuples & Dynamic Feedback | arXiv 2024 | Phase 5 |
| Planning with Consistency Models for Model-Based Offline RL | TMLR | Phase 6 |

---

## 🧪 실험 대시보드

모든 구현 실험의 학습 로그, 하이퍼파라미터, 결과 시각화는 wandb에서 확인하실 수 있습니다.

**[→ W&B 대시보드 전체 보기](https://wandb.ai/YOUR_USERNAME/portfolio-tsuruoka-lab)**

---

## 🔧 환경

```bash
Python         3.10+
PyTorch        2.x
Lightning      2.x
NumPy          1.24+
wandb          0.16+
```

```bash
pip install -r requirements.txt
```

---

## 📬 Contact

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- W&B: [YOUR_USERNAME](https://wandb.ai/YOUR_USERNAME)
