import numpy as np 
A = np.array([[1,2,3],[4,5,6]]) # shape (2,3)
B = np.array([[7,8],[9,10],[11,12]]) # shape (3,2)
m, k = A.shape
k2, n = B.shape
C = np.zeros((m,n)) # shape (2,2)

for i in range(m): 
    for j in range(n): 
        for p in range(k): 
            C[i][j] += A[i][p] * B[p][j] # 내적  한 칸씩
print(C) # [[ 58. 64.] # [139. 154.]]