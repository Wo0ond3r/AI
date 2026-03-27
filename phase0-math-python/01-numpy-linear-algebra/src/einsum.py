import numpy as np 
A = np.array([[1,2,3],[4,5,6]]) 
B = np.array([[7,8],[9,10],[11,12]]) # einsum — 첨자로 연산을 기술 # "ik,kj->ij" : A[i,k] × B[k,j] 를 k에 대해 합산 → C[i,j] 
C = np.einsum('ik,kj->ij', A, B) 
print(C) # [[ 58 64], [139 154]] 
# Transformer 코드에서 실제로 보이는 einsum들: 
# # 'bhqd,bhkd->bhqk' (Attention score 계산) 
# # 'bld,dm->blm' (선형 변환) 
# # 'bic,bjc->bij' (배치 내적)