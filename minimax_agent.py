import math
import copy

def check_winner(board, player):
    """현재 보드에서 특정 플레이어가 승리했는지 확인 (환경 코드와 동일한 로직)"""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # 가로
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # 세로
        [0, 4, 8], [2, 4, 6]             # 대각선
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

def minimax(board, depth, is_maximizing, current_player):
    """
    미래를 시뮬레이션하는 재귀 함수입니다.
    - is_maximizing이 True면 내 턴 (가장 큰 값 선택)
    - False면 상대 턴 (가장 작은 값 선택)
    """
    opponent = -1 * current_player

    # 1. 터미널 상태 (게임 종료) 판정 및 점수 반환 (당신이 생각한 R 값)
    if check_winner(board, current_player):
        return 1  # 내가 이기면 +1
    if check_winner(board, opponent):
        return -1 # 상대가 이기면 -1
    if 0 not in board:
        return 0  # 빈칸이 없으면 무승부 0

    valid_actions = [i for i, value in enumerate(board) if value == 0]

    # 2. 내 턴 (Maximizer) : 가장 높은 점수를 찾는다
    if is_maximizing:
        best_score = -math.inf # 아주 작은 값으로 초기화
        for action in valid_actions:
            # 상상 속에서 돌을 둬본다
            board[action] = current_player 
            # 상대방의 턴으로 넘겨 미래를 시뮬레이션 (재귀 호출)
            score = minimax(board, depth + 1, False, current_player)
            # 상상을 끝내고 돌을 다시 치운다 (원상복구)
            board[action] = 0 
            # 더 좋은 점수가 나오면 갱신
            best_score = max(best_score, score)
        return best_score

    # 3. 상대 턴 (Minimizer) : 나에게 가장 치명적인(낮은) 점수를 찾는다
    else:
        best_score = math.inf # 아주 큰 값으로 초기화
        for action in valid_actions:
            board[action] = opponent
            score = minimax(board, depth + 1, True, current_player)
            board[action] = 0
            best_score = min(best_score, score)
        return best_score

def get_best_action(env):
    """현재 환경에서 Minimax 알고리즘을 사용해 최적의 수를 찾습니다."""
    best_score = -math.inf
    best_action = None
    
    valid_actions = [i for i, value in enumerate(env.board) if value == 0]
    
    for action in valid_actions:
        # 실제 환경을 망가뜨리지 않기 위해 보드 복사본 사용
        board_copy = copy.deepcopy(env.board)
        board_copy[action] = env.current_player
        
        # 첫 번째 수는 내가 두었으니, 다음은 상대방 턴(is_maximizing=False)으로 시뮬레이션 시작
        score = minimax(board_copy, 0, False, env.current_player)
        
        # 당신의 2번 답변: "최대한의 값을 선택하는게 맞는거 같아" -> 여기서 적용됨
        if score > best_score:
            best_score = score
            best_action = action
            
    return best_action