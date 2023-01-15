# Sum 노드 구현

# 계산 그래프에서 Sum 노드는 D 길이의 벡터 N개를 각 원소 별로 더한 총합을
# 각 원소의 값으로 하는 D 길이의 벡터 1개를 만드는 노드

import numpy as np

# 벡터의 길이 D를 8로, 벡터의 개수 N을 7로 설정
D, N = 8, 7

# 순전파 입력값 x : -1 ~ 1 사이의 실수 값이 각 원소로 저장된 N행 D열 형상의 행렬
x = np.random.randn(N, D)
# 순전파 출력값 y : x 행렬을 0축(axis) 방향으로 합하고 2차원 행렬의 차원 수 유지
y = np.sum(x, axis=0, keepdims=True)

# 역전파 입력값 dy : -1 ~ 1 사이의 실수 값이 각 원소로 저장된 1행 D열 형상의 행렬(벡터)
dy = np.random.randn(1, D)
# 역전파 출력값 dx : dy 행렬(벡터)를 N개 복제하여 N개의 행 dy 벡터 길이만큼의 열을 가진 행렬 (N행 D열 형상의 행렬)
dx = np.repeat(dy, N, axis=0)

# repeat 노드의 순전파 계산은 sum 노드의 역전파 계산과 같고
# repeat 노드의 역전파 계산은 sum 노드의 순전파 계산과 같음
# 즉 repeat 노드와 sum 노드는 반대관계