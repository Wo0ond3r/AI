import numpy as np 
A = np.array([[1,2,3],[4,5,6]]) 
B = np.array([[7,8],[9,10],[11,12]]) # 세 가지 표기 — 전부 동일한 결과 
C1 = np.dot(A, B) # 고전적인 방법 
C2 = A @ B # Python 3.5+ 연산자 (논문 코드에서 가장 흔함) 
C3 = np.matmul(A, B) # 명시적 방법 
print(C1) # [[ 58 64], [139 154]] 
print(np.allclose(C1, C2, C3)) # True # shape 확인 습관 — 항상 찍어보세요 
print(A.shape, B.shape, C1.shape) # (2, 3) (3, 2) (2, 2)