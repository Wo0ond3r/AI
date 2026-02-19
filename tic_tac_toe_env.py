import numpy as np

class TicTacToeEnv:
    def __init__(self):
        #state (S): 3x3 보드를 1차원 배열 (크기 9)로 표현
        # 0: 빈 칸, 1: 플레이어 1 (나), -1: 플레이어 2 (상대)
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1 
        self.done = False # 게임 종료 여부
    
    def reset(self):
        """
        게임을 초기화하고 첫 상태(S_0)를 반환합니다.
        """
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.done = False
        return self.board.copy() # 초기 상태 (S_0)
    
    def step(self, action):
        """
        행동(A)을 받아 상태를 변화(P)시키고, 보상(R)을 반환하는 핵심 함수입니다.
        """
        # 에러 처리: 이미 끝난 게임이거나, 돌이 있는 곳에 두려고 할 때
        if self.done or self.board[action] != 0:
            return self.board.copy(), -10, True, {"msg": "잘못된 행동"}
        
        # 1. 상태 변이 (Transitionm P): 보드에 돌을 놓는다
        self.board[action] = self.current_player

        # 2. 보상 계산 (Reward R) 및 종료 판정
        reward = 0
        if self.check_winner(self.current_player):
            reward = 1  # 승리 보상
            self.done = True
        elif np.all(self.board != 0):
            reward = 0.5  # 무승부 보상
            self.done = True
        else:
            reward = 0  # 게임 진행 중 보상
            self.done = False

        # 턴 넘기기
        if not self.done:
            self.current_player *= -1

        # 다음 상태(S_{t+1}), 보상(R_{t+1}), 종료여부 반환
        return self.board.copy(), reward, self.done, {}
    
    def check_winner(self, player):
        """ 승리 조건을 확인하는 내부 함수 (가로, 세로, 대각선) """
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # 가로
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # 세로
            [0, 4, 8], [2, 4, 6]             # 대각선
        ]
        for condition in win_conditions:
            if all(self.board[i] == player for i in condition):
                return True
        return False
    
    def render(self):
        """현재 상태(S)를 화면에 출력 (디버깅 용도)"""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n")
        for i in range(3):
            row = [symbols[self.board[i*3 + j]] for j in range(3)]
            print(" " + " | ".join(row))
            if i < 2:
                print("---+---+---")
        print("\n")