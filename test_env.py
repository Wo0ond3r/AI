import torch
import wandb
import random

# 1. WandB 초기화 (프로젝트 이름 설정)
wandb.init(project="game-ai-portfolio", name="env-test-run")

# 설정값 저장 (Hyperparameters)
config = wandb.config
config.learning_rate = 0.01
config.epochs = 10

print(f"CUDA Available: {torch.cuda.is_available()}")

# 2. 가상의 학습 루프 (Mock Training)
for epoch in range(config.epochs):
    # 가상의 Loss와 Accuracy 생성
    loss = 1.0 / (epoch + 1) + random.random() * 0.1
    accuracy = 0.6 + (epoch * 0.03)
    
    # 3. WandB에 로그 기록
    wandb.log({"loss": loss, "accuracy": accuracy})
    print(f"Epoch {epoch}: Loss {loss:.4f}, Acc {accuracy:.4f}")

print("Test Completed!")
wandb.finish()