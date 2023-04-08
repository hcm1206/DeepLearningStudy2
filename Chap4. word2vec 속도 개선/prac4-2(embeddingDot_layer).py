# word2vec 속도 개선 방법 2
# 어휘 수가 거대해지면 은닉층->출력층 가중치 행렬 Wout 간 행렬곱 계산이 매우 복잡해짐
# Softmax 계층의 계산도 어휘 수가 많아질 수록 exp 계산을 그만큼 많이 해야 함

# 위 문제를 해결하기 위해 다중 분류를 이진 분류로 근사하여 문제를 단순화하는 네거티브 샘플링 기법 활용
# 다중 분류 : N개의 단어 중 타깃 단어로 가장 알맞은 단어는 무엇인가?
# 이진 분류 : 타깃 단어로 N번째 단어가 가장 적절한가?(Yes/No)

# 1. 출력층까지의 가중치 행렬 단순화

# 이진 분류 방식으로 바꾸면 은늑칭과 가중치(Wout) 행렬의 내적(행렬곱)은 타깃단어에 해당하는 열(단어 벡터)만 추출
# 출력층은 추출된 단어 벡터에 대한 점수 값 하나 만을 출력
# 이전까지의 출력층 계산은 모든 단어를 대상으로 계산을 수행했지만 이번 구현에서는 특정 단어 하나에만 주목하여 계산 수행

# 2. 출력층에서의 확률 계산을 단순화

# 출력층에서 예상 타깃 단어에 대한 점수를 계산했다면 이 점수에 시그모이드 함수를 적용하여 확률로 변환
# 이후 이 확률을 교차 엔트로피 오차를 이용하여 실제 정답값과 비교하여 손실을 구한 후 신경망 학습에 적용

# 시그모이드 함수에 입력값을 적용하면 출력값은 0~1 사이 실수 형태의 확률 형식을 구할 수 있고, 
# 이 확률 값을 교차 엔트로피 오차에 정답 레이블과 함께 입력하면 (신경망에서 예측한)확률 값과 실제 정답을 비교하여 그 차이를 출
# 정답 레이블을 0(거짓) 또는 1(참) 중 하나의 값이며, 높은 확률의 예측 값과 실제 정답 값이 일치하다면 손실(오차) 값이 작아짐
# 이 손실(오차) 값이 작으면 역전파에 흘러가서 신경망이 '작게' 학습하고, 손실(오차) 값이 크다면 역전파에 흘러가서 신경망에 '크게' 학습


# 예측한 값을 Embedding 계산을 통해 하나의 단어 벡터를 출력하여 Wout 가중치를 획득하는 부분(Emded)과
# Wout 가중치와 은닉층을 계산하는 부분(dot)를 통합하여 Embedding Dot 계층 도입
# Embedding Dot 계층 구현

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
# prac 4-1에서 구현했던 Embedding 계층 불러옴
from common.layers import Embedding

# 출력층 계산을 위한 Embedding(타깃 단어 벡터를 구하는 과정) 계산과 Dot(타깃 단어 벡터와 은닉층과의 벡터 내적(곱)) 계산을 통합한 EmbeddingDot 클래스 구현
class EmbeddingDot:
    # 클래스 생성자(인자로 가중치 행렬 W 입력받음)
    def __init__(self, W):
        # Embedding 계층 객체에 인자로 W를 넣어 맴버 변수 embed로 저장
        self.embed = Embedding(W)
        # embed 계층의 매개변수를 맴버 변수 params로 저장
        self.params = self.embed.params
        # embed 계층의 기울기를 맴버 변수 grads로 저장
        self.grads = self.embed.grads
        # 맴버 변수로 순전파 계산 결과를 저장할 cache를 선언하고 초기화하지 않음
        self.cache = None
    
    # EmbeddingDot 계층 순전파 계산 메소드(인자로 은닉층 h, 입력 단어(들)의 인덱스 행렬 idx 입력받음)
    def forward(self, h, idx):
        # embed 계층에 입력 단어의 인덱스(idx)를 입력하여 순전파 계산한 결과(단어 벡터)를 target_W에 저장 (Embedding 계산)
        target_W = self.embed.forward(idx)
        # 입력 단어 벡터(target_W)와 은닉층의 값들을 1차원 축(행)을 기준으로 스칼라 곱한 결과들의 합을 out에 저장 (Dot 계산)
        out = np.sum(target_W * h, axis=1)

        # 은닉층과 입력 단어 벡터를 클래스 맴버변수 cache에 저장
        self.cache = (h, target_W)
        # 계산된 out 값을 리턴
        return out
    
    # EmbeddingDot 계층 역전파 계산 메소드(인자로 역전파 입력값 dout 입력받음)
    def backward(self, dout):
        # 맴버변수 cache를 통해 순전파 계산 시 사용했던 은닉층 h와 입력 단어 벡터 target_W 저장
        h, target_W = self.cache
        # 역전파 입력값 행렬 dout의 열을 1차원으로 재배열하여 저장
        dout = dout.reshape(dout.shape[0], 1)

        # dtarget_W에 재배열된 dout과 은닉층 h의 스칼라 곱 결과 저장
        dtarget_W = dout * h
        # dtarget_W 값을 Embedding 계층 역전파 계산에 입력
        self.embed.backward(dtarget_W)
        # dout과 target_W 값의 곱을 계산하여 dh로 저장
        dh = dout * target_W
        # dh 값 리턴
        return dh
    

    # 변수 예시 :
    # 매개변수 W = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20]]
    # 입력 단어 인덱스 idx = [0, 3, 1] (3개 단어 각각의 인덱스(미니배치))
    # Embedding 계층 계산으로 구한 입력단어(들)의 단어 벡터 target_W = [[0, 1, 2], [9, 10, 11], [3, 4, 5]]
    # 은닉층 h = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # 행렬 내적(곱) target_W * h = [[0, 1, 4], [27, 40, 55], [18, 28, 40]]
    # 순전파 출력값 out(target_W * h 행렬의 각 행에 대한(axis=1) 계산) = [5, 122, 86]