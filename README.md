# 🎓 Portfolio — Tsuruoka Lab (Tokyo University)

> **목표:** 도쿄대학 Tsuruoka 연구실 (자연언어처리 · 강화학습 · 게임 AI) 진학을 위한 연구 포트폴리오
>
> **방향:** 기초 딥러닝부터 LLM × Game AI까지, 논문 구현 중심으로 구성

[![wandb](https://img.shields.io/badge/Experiments-W%26B-yellow?logo=weightsandbiases)](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🗺️ 전체 커리큘럼 로드맵

```
Phase 0  수학 & Python 기초          ████████░░  진행 중
Phase 1  Deep Learning 기초          ░░░░░░░░░░  예정
Phase 2  NLP & Transformers          ░░░░░░░░░░  예정
Phase 3  Large Language Models       ░░░░░░░░░░  예정
Phase 4  RL & Game AI                ░░░░░░░░░░  예정
Phase 5  LLM × Games  ★ 핵심        ░░░░░░░░░░  예정
Phase 6  SOTA & 독자적 연구           ░░░░░░░░░░  예정
```

---

## 📁 프로젝트 구조

```
portfolio-tsuruoka-lab/
│
├── phase0-math-python/
│   ├── 01-numpy-linear-algebra/     ✅ 완료
│   ├── 02-micrograd-from-scratch/   🔄 진행 중
│   └── 03-mlp-mnist/                ⬜ 예정
│
├── phase1-deep-learning/
│   ├── 01-resnet-from-scratch/      ⬜ 예정
│   └── ...
│
├── phase2-nlp-transformers/
│   ├── 01-transformer-from-scratch/ ⬜ 예정
│   ├── 02-bert-pretrain/            ⬜ 예정
│   └── ...
│
├── phase3-llm/
│   ├── 01-rlhf-minimal/             ⬜ 예정
│   └── ...
│
├── phase4-rl-game-ai/
│   ├── 01-dqn-atari/                ⬜ 예정
│   ├── 02-alphazero-connect4/       ⬜ 예정
│   └── ...
│
├── phase5-llm-x-games/              ★ 핵심 목표
│   ├── 01-react-agent/              ⬜ 예정
│   ├── 02-tot-game-planning/        ⬜ 예정
│   ├── 03-mahjong-value-net/        ⬜ 예정  (Tsuruoka Lab 2024)
│   └── 04-llm-reasoning-tuning/     ⬜ 예정  (Tsuruoka Lab 2024)
│
└── phase6-research/                 ★ 독자적 연구
    └── ...
```

---

## 🧪 실험 대시보드 (W&B)

모든 구현 실험의 학습 로그, 하이퍼파라미터, 결과 시각화는 아래에서 확인할 수 있습니다.

**[→ W&B 대시보드 보기](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab)**

| Phase        | 실험명                      | W&B 링크                                                                                            |
| ------------ | --------------------------- | --------------------------------------------------------------------------------------------------- |
| Phase 0 - 01 | NumPy Linear Algebra Basics | [링크](https://wandb.ai/swsw778-korea-university/portfolio-tsuruoka-lab/workspace?nw=nwuserswsw778) |
| Phase 0 - 02 | Micrograd Autograd Engine   | 진행 예정                                                                                           |
| Phase 0 - 03 | MLP MNIST From Scratch      | 진행 예정                                                                                           |

---

## 🎯 목표 연구실

**[Tsuruoka Laboratory](https://www.logos.t.u-tokyo.ac.jp/)** — 도쿄대학 대학원 정보이공학계연구과

> _"NLP · Reinforcement Learning · Artificial Intelligence for Games"_
> — Yoshimasa Tsuruoka

### 관련 논문 (구현 대상)

| 논문                                                              | 발표       | 구현 상태  |
| ----------------------------------------------------------------- | ---------- | ---------- |
| MJ-DLVAT: Deep Learning Value Assessment for Mahjong              | CoG 2024   | ⬜ Phase 5 |
| Improving Arithmetic Reasoning via Relation Tuples & Verification | arXiv 2024 | ⬜ Phase 5 |
| Planning with Consistency Models for Offline RL                   | TMLR       | ⬜ Phase 6 |

---

## 🔧 환경

```bash
Python  3.10+
PyTorch 2.x
NumPy   1.24+
wandb   0.16+
```

```bash
# 환경 설치
pip install -r requirements.txt
```

---

## 📬 Contact

- GitHub: [@Wo0ond3r](https://github.com/Wo0ond3r)
- W&B: [@swsw778-korea-university](https://wandb.ai/swsw778-korea-university)
