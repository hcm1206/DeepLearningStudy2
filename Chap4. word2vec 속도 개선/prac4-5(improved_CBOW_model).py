# prac 3-5에서 구현한 simpleCBOW 모델을 기반으로 Embedding 계층과 Negative Sampling Loss 계층을 적용하여 개선한 모델 구현
# 추가로 맥락 윈도우 크기를 조절 가능하도록 기능 확장

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.layers import Embedding
from files.negative_sampling_layer import NegativeSamplingLoss

# CBOW 클래스 정의
class CBOW:
    # 클래스 초기화(어휘 수, 은닉층 개수, 맥락 윈도우 크기, 말뭉치 ID 입력받음)
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        # 어휘 수와 은닉층 개수를 각각 V, H로 저장
        V, H = vocab_size, hidden_size 

        # 입력층->은닉층 계산에 필요한 V×H 형상의 랜덤 실수 매개변수(입력층 측 가중치)를 W_in으로 저장
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        # 은닉층->출력층 계산에 필요한 V×H 형상의 랜덤 실수 매개변수(출력층 측 가중치)를 W_out으로 저장
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # CBOW 모델의 Embedding 계층을 저장할 빈 리스트 생성하여 맴버변수 in_layers로 저장
        self.in_layers = []
        # 반복 변수 i를 이용하여 맥락 크기 × 2 (맥락 단어 크기)만큼 반복
        for i in range(2 * window_size):
            # 입력층 측 가중치(단어 벡터)를 저장하는 Embedding 계층을 생성하여 layer로 저장
            layer = Embedding(W_in)
            # 생성한 Embedding 계층 layer를 맴버변수 in_layers로 저장 (맥락 범위 내의 단어 수만큼 Embedding 계층 생성하여 저장)
            self.in_layers.append(layer)
        # NegativeSamplingLoss 계층 클래스를 생성하고 출력층 측 가중치, 말뭉치 ID, 제곱값(0.75), 샘플링 값 개수(5) 입력하여 맴버변수 ns_loss로 저장
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # layers를 선언하여 모든 Embedding 계층과 (2차원화시킨) NegativeSamplingLoss 계층을 layers에 모아서 저장
        layers = self.in_layers + [self.ns_loss]
        # 가중치 매개변수와 기울기를 저장할 빈 리스트를 생성하여 맴버변수 params, grads로 저장
        self.params, self.grads = [], []
        # 모든 계층(layers)에서 계층(layer)을 하나씩 뽑아서 반복
        for layer in layers:
            # params에 현재 계층 layer의 가중치 매개변수 추가하여 저장
            self.params += layer.params
            # grads에 현재 계층 layer의 기울기 추가하여 저장
            self.grads += layer.grads

        # 입력층 가중치(W_in)를 단어 분산 표현 벡터(word_vecs)로 사용
        self.word_vecs = W_in

    # 순전파 메소드 구현(맥락 단어, 타깃 단어 입력)
    def forward(self, contexts, target):
        # 은닉층을 계산하여 저장할 변수 h를 선언하고 0으로 초기화
        h = 0
        # 모델의 모든 계층이 저장된 맴버변수 in_layers의 모든 계층을 반복하며 인덱스를 i로, 계층을 layer로 설정
        for i, layer in enumerate(self.in_layers):
            # i번째 인덱스의 모든 맥락 단어를 현재 계층의 순전파 함수(forward())에 입력한 결과를 은닉층 h에 더하여 저장
            h += layer.forward(contexts[:, i])
            # 은닉층의 값을 기존 은닉층에 저장되어있던 값을 모든 계층의 개수로 나눈 값으로 저장(은닉층 행렬의 평균값을 저장하는 것)
            h *= 1 / len(self.in_layers)
            # 은닉층 결과 h와 타깃값을 네거티브 샘플링 계층 ns_loss의 순전파 함수로 입력하여 손실(loss)값을 구함
            loss = self.ns_loss.forward(h, target)
            # 구한 손실값 loss 리턴
            return loss
    
    # 역전파 메소드 구현(역전파 계산값(기본값 1) 입력)
    def backward(self, dout=1):
        # 역전파 계산값을 네거티브 샘플링(ns_loss)의 역전파 함수에 입력하여 나온 결과를 역전파 계산값으로 저장
        dout = self.ns_loss.backward(dout)
        # 역전파 계산값을 Embedding 계층의 수로 나누어 평균을 구하고 그 평균값으로 저장
        dout *= 1 / len(self.in_layers)
        # 구한 역전파 계산값 평균을 다시 각 Embedding 계층의 역전파 메소드에 입력
        for layer in self.in_layers:
            layer.backward(dout)
        # 리턴값 없음
        return None
    


# 위 CBOW 클래스 코드는 files/cbow.py 파일로 저장