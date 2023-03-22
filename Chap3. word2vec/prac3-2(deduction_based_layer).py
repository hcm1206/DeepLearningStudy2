# prac3-1의 신경망에서 사용된 행렬곱을 Chap1의 prac1-9에서 구현한 MatMul 계층으로 수행

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.layers import MatMul

# 인덱스가 0번인 'you' 단어의 원-핫 벡터 (2차원 미니 배치로 포함)
c = np.array([[1,0,0,0,0,0,0]])
# 7×3 형상의 랜덤 0~1 사이 실수가 저장된 행렬을 가중치 W로 저장
W = np.random.randn(7,3)
# 가중치 행렬 W를 행렬곱을 계산하는 완전연결 Matmul 계층에 입력하여 layer로 저장
layer = MatMul(W)
# Matmul 계층(layer)에 입력값 행렬 c를 입력하여 순전파 계산한 결과를 은닉층으로 h에 저장
h = layer.forward(c)
# 은닉층 h 출력
print(h)