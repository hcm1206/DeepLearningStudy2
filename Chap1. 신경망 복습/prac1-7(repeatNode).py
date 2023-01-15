# Repeat 노드 구현

# 계산 그래프에서 Repeat 노드는 분기 노드의 연장선으로, 
# 어떤 벡터(1차원 행렬) D를 같은 값으로 N개 복제하여 2차원 행렬로 만드는 노드

# Repeat 노드의 역전파는 상류에서의 기울기의 합

import numpy as np

# 벡터의 크기 D를 8, 복제(반복)할 벡터 수 N을 7로 설정
D, N = 8, 7

# 순전파 입력값 x : -1 ~ 1 사이의 랜덤 실수 값을 D(8) 크기의 벡터 요소에 각각 저장
x = np.random.randn(1, D)
# 순전파 출력값 y : x 벡터를 N개 복제하여 0축(axis) 방향으로 복제 (N행 × D열 크기의 행렬이 됨)
y = np.repeat(x, N, axis=0)

# 역전파 입력값 dy : 각 요소에 -1 ~ 1 사이의 랜덤 실수 값이 저장된 N × D 크기의 행렬
dy = np.random.randn(N, D)
# 역전파 출력값 dx : dy 행렬을 0축(axis) 방향으로 합하고 2차원 행렬의 차원 수 유지
dx = np.sum(dy, axis=0, keepdims = True)