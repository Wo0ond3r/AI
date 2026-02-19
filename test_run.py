import random
from tic_tac_toe_env import TicTacToeEnv
from minimax_agent import get_best_action

def get_random_action(env):
    """무작위 에이전트의 행동 선택"""
    valid_actions = [i for i, value in enumerate(env.board) if value == 0]
    return random.choice(valid_actions)

def play_game(render=False):
    """한 판의 게임을 진행하고 결과를 반환합니다."""
    env = TicTacToeEnv()
    env.reset()
    
    while not env.done:
        # 1. 플레이어 1 (Minimax AI)의 턴
        if env.current_player == 1:
            action = get_best_action(env)
        # 2. 플레이어 -1 (무작위 AI)의 턴
        else:
            action = get_random_action(env)
            
        _, reward, _, _ = env.step(action)
        
        if render:
            env.render()
            
    # 게임 종료 후 승자 판정 (환경 코드의 check_winner 활용)
    if env.check_winner(1):
        return 1   # Minimax 승리
    elif env.check_winner(-1):
        return -1  # 무작위 AI 승리
    else:
        return 0   # 무승부

def main():
    print("=== 100판 시뮬레이션 시작 (Minimax vs Random) ===")
    
    results = {'minimax_wins': 0, 'random_wins': 0, 'draws': 0}
    num_games = 100
    
    for i in range(num_games):
        # 마지막 판(100번째)만 어떻게 이기는지 과정을 화면에 출력
        render = True if i == (num_games - 1) else False 
        
        winner = play_game(render=render)
        
        if winner == 1:
            results['minimax_wins'] += 1
        elif winner == -1:
            results['random_wins'] += 1
        else:
            results['draws'] += 1
            
    print("\n=== 최종 결과 ===")
    print(f"Minimax AI 승리: {results['minimax_wins']}회")
    print(f"무작위 AI 승리 : {results['random_wins']}회 (단 한 번이라도 이기면 알고리즘 실패!)")
    print(f"무승부         : {results['draws']}회")

if __name__ == "__main__":
    main()