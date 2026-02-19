import random
import time
# 방금 만든 환경 클래스를 불러옵니다.
from tic_tac_toe_env import TicTacToeEnv

def main():
    # 1. 환경(우주) 생성 및 초기화 (S_0)
    env = TicTacToeEnv()
    state = env.reset()
    
    print("=== 틱택토 환경 시뮬레이션 시작 ===")
    env.render()

    step_count = 0
    # 게임이 끝날 때(env.done == True)까지 턴을 반복합니다.
    while not env.done:
        # 2. 현재 상태에서 둘 수 있는 빈칸(Valid Actions) 찾기
        # 보드(state) 배열에서 값이 0(빈칸)인 인덱스만 리스트로 뽑아냅니다.
        valid_actions = [i for i, value in enumerate(env.board) if value == 0]
        
        # 3. 무작위 행동 선택 (Random Policy)
        action = random.choice(valid_actions)
        
        player_symbol = 'O' if env.current_player == 1 else 'X'
        print(f"턴 {step_count + 1}: 플레이어 {player_symbol}({env.current_player})가 위치 {action}에 돌을 둡니다.")
        
        # 4. 환경에 행동(A_t) 전달하여 상태를 변화(Step)
        next_state, reward, done, info = env.step(action)
        env.render()
        
        step_count += 1
        time.sleep(0.5) # 눈으로 진행 상황을 쫓을 수 있도록 0.5초 대기

    # 5. 게임 종료 후 결과 판정 (R_T)
    print("=== 게임 종료 ===")
    if reward == 1:
        # step() 내부에서 턴이 넘어가버렸으므로, 승리자는 방금 돌을 둔 이전 플레이어입니다.
        winner = -1 * env.current_player 
        winner_symbol = 'O' if winner == 1 else 'X'
        print(f"결과: 플레이어 {winner_symbol}({winner}) 승리!")
    else:
        print("결과: 무승부!")

if __name__ == "__main__":
    main()