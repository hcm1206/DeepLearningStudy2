# 간단한 CBOW 모델 구현
# MatMul 계층, SoftmaxWithLoss 계층은 prac1-9, prac1-12에서 구현된 것과 같음

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

# 간단하게 구현할 CBOW 신경망 모델 클래스 정의
class SimpleCBOW:
    # 생성자 인자로 vocab_size(어휘 숫자), hidden_size(은닉층 개수) 입력받음(신경망 기본 세팅)
    def __init__(self, vocab_size, hidden_size):
        # 입력받은 어휘 수와 은닉층 개수를 각각 V와 H로 저장
        V, H = vocab_size, hidden_size

        # 입력층->은닉층을 계산하기 위한 W_in 매개변수 생성 (V×H 크기의 랜덤 실수가 저장된 행렬)
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        # 은닉층->출력층을 계산하기 위한 W_out 매개변수 생성 (H×V 크기의 랜덤 실수가 저장된 행렬)
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 입력값의 첫번째 단어 벡터와 W_in 가중치 매개변수를 행렬곱하기 위한 MatMul 객체를 in_layer0로 저장
        self.in_layer0 = MatMul(W_in)
        # 입력값의 두번째 단어 벡터와 W_in 가중치 매개변수를 행렬곱하기 위한 MatMul 객체를 in_layer1로 저장
        self.in_layer1 = MatMul(W_in)
        # 은닉층과 W_out 가중치 매개변수를 행렬곱하기 위한 MatMul 객체를 out_layer로 저장
        self.out_layer = MatMul(W_out)
        # 소프트맥스 함수와 교차 엔트로피 오차를 계산하기 위한 SoftmaxWithLoss 객체를 loss_layer로 저장
        self.loss_layer = SoftmaxWithLoss()

        # 행렬곱이 진행되는 3가지 계층을 layers로 저장
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        # 가중치 매개변수와 기울기를 저장할 빈 리스트 2개 생성
        self.params, self.grads = [], []
        # 3가지 행렬곱 계층을 하나씩 layer로 지정하며 반복
        for layer in layers:
            # 모델의 매개변수를 저장하는 리스트에 해당 계층의 매개변수를 추가하여 저장
            self.params += layer.params
            # 모델의 기울기를 저장하는 리스트에 해당 계층의 기울기를 추가하여 저장
            self.grads += layer.grads
        
        # 입력값과 행렬곱되는 W_in 기울기 매개변수를 단어 벡터로 지정하여 word_vecs로 저장
        self.word_vecs = W_in


        # 위 계층에서 MatMul 계층은 맥락 단어 범위인 window_size에 포함되는 단어 개수만큼 만들어야 함
        # 예시에서는 맥락 단어 범위에 들어가는 단어가 2개 이므로 입력값을 위한 2개의 MatMul 계층이 존재

        # 이 코드에서는 여러 계층에서 같은 가중치를 공유하여 params 리스트에 중복된 가중치가 여러 개 존재
        # 이 경우 Adam이나 Momentum 등의 매개변수 갱신 처리가 기존과 달라지기 때문에 Trainer 클래스에서 매개변수 중복을 없애는 작업을 처리

        # 신경망 순전파 계산 메소드 (맥락 행렬, 타깃 행렬 입력받음)
        def forward(self, contexts, target):
            # 1번째 맥락 단어 입력값(contexts 행렬의 0번째 행)을 in_layer0(입력층 MatMul) 계층 순전파 계산에 입력한 결과를 h0 은닉층으로 저장
            h0 = self.in_layer0.forward(contexts[:,0])
            # 2번째 맥락 단어 입력값(contexts) 행렬의 1번째 행)을 in_layer1(입력층 MatMul) 계층 순전파 계산에 입력한 결과를 h1 은닉층으로 저장
            h1 = self.in_layer1.forward(contexts[:,1])
            # 행렬곱 계산이 완료된 각 맥락 단어 은닉층 행렬의 평균값을 구하여 h로 저장
            h = (h0 + h1) * 0.5
            # 은닉층 h를 out_layer(출력층 MatMul) 계층 순전파 계산에 입력한 결과를 score로 저장
            score = self.out_layer.forward(h)
            # 최종 행렬곱 계산 결과 score와 타깃값 원-핫 벡터 target을 loss_layer(소프트맥스+오차 엔트로피 교차 연산) 계층에 입력한 순전파 계산 결과(손실값)를 loss로 저장
            loss = self.loss_layer.forward(score, target)
            # 계산된 손실값 loss 리턴
            return loss
        
        # 입력된 contexts는 3차원 넘파이 배열로, 
        # 0차원 원소 수 : 미니배치 원소 수
        # 1차원 원소 수 : 맥락 단어 범위(윈도우 크기)
        # 2차원 원소 수 : 원-핫 벡터 크기(어휘 개수)

        # 신경망 역전파 계산 메소드 (역전파 입력값(기본값 1) 입력받음)
        def backward(self, dout=1):
            # 최초 역전파 입력값 dout에 대한 loss_layer(소프트맥스+오차 엔트로피 교차) 계층 역전파 계산을 ds에 저장
            ds = self.loss_layer.backward(dout)
            # 역전파 계산값 ds에 대한 out_layer(출력층 MatMul) 계층 역전파 계산에 입력한 결과를 da로 저장
            da = self.out_layer.backward(ds)
            # da에 대해 평균값 계산 과정 역전파 계산 (곱셈은 순전파의 입력을 서로 바꿔 기울기에 곱하고, 덧셈은 각 입력값마다 그대로 통과해서 보냄)
            da *= 0.5
            # da에 대한 in_layer1(2번째 입력층 MatMul) 계층 역전파 계산
            self.in_layer1.backward(da)
            # da에 대한 in_layer0(1번째 입력층 MatMul) 계층 역전파 계산
            self.in_layer0.backward(da)
            # 리턴은 없음(역전파 출력값은 필요없고 역전파 계산을 통한 기울기 grads 획득이 목표이므로)
            return None
        
        # 순전파 계산과 역전파 계산 시 각 계층(MatMul, SoftmaxWithLayer 등)에서 자체적으로 순전파, 역전파 메서드와 가중치, 기울기 매개변수를 포함하고 갱신함


# 이 코드는 files/simple_cbow.py 파일에 동일한 내용으로 구현되어 있음