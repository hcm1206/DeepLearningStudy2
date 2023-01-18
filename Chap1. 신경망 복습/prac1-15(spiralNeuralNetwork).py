# 은닉층이 하나인 신경망 구현

import sys, os
# 상위 디렉토리 파일 접근을 위한 코드
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
# Affine, Sigmoid, SoftmaxWithLoss 계층 코드를 common.layers 파일에서 불러옴
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

# 2층 신경망 클래스 구현
class TwoLayerNet:
    # 클래스 생성자(입력값 크기, 은닉층 크기, 출력층 크기 입력)
    def __init__(self, input_size, hidden_size, output_size):
        # 입력받은 값들을 변수로 저장
        I, H, O = input_size, hidden_size, output_size

        # 첫번째 가중치 매개변수를 I×H 형상의 랜덤값 행렬로 저장
        W1 = 0.01 * np.random.randn(I, H)
        # 첫번째 편향 매개변수를 H 행렬과 같은 형상의 0으로 이루어진 행렬로 저장
        b1 = np.zeros(H)
        # 두번째 가중치 매개변수를 H×O 형상의 랜덤값 행렬로 저장
        W2 = 0.01 * np.random.randn(H, O)
        # 두번째 편향 매개변수를 O 행렬과 같은 형상의 0으로 이루어진 행렬로 저장
        b2 = np.zeros(O)

        # 첫번째 가중치, 편향 매개변수를 갖는 Affine 계층, 시그모이드 계층, 두번쨰 가중치, 편향 매개변수를 갖는 Affine 계층 순으로 계층을 클래스 리스트 변수로 저장
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        # 결과를 출력하기 위한 손실함수가 포함된 SoftmaxWithLoss 계층을 클래스 변수로 저장
        self.loss_layer = SoftmaxWithLoss

        # 신경망의 모든 매개변수와 기울기를 저장할 빈 리스트 생성
        self.params, self.grads = [], []
        # 각 계층 별로 반복하여 매개변수와 기울기를 불러와 리스트에 각각 저장
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # 데이터 추론 메서드 구현(입력값 x 입력)
    def predict(self, x):
        # 각 계층 순서대로 반복
        for layer in self.layers:
            # 해당 계층의 순전파 계산 진행
            x = layer.forward(x)
        # 모든 순전파 계산 후 최종값 리턴
        return x

    # 데이터 학습용 순전파 메서드 구현 (입력값 x, 정답 레이블 t 입력)
    def forward(self, x, t):
        # 데이터 추론 메서드를 이용하여 입력값 x의 순전파 계산 진행 후 결과를 score로 저장
        score = self.predict(x)
        # 추론 결과를 SoftmaxWithLoss 계층을 이용하여 실제 정답 데이터와 비교 후 손실 값을 loss에 저장
        loss = self.loss_layer.forward(score, t)
        # 손실값 loss 리턴
        return loss

    # 매개변수 갱신용 역전파 메서드 구현 (역전파 입력값 dout(기본값 1) 입력)
    def backward(self, dout=1):
        # 먼저 SoftmaxWithLoss 계층에서의 역전파 계산 진행
        dout = self.loss_layer.backward(dout)
        # 이후 기존 계층의 역순으로 계층마다 반복
        for layer in reversed(self.layers):
            # 각 계층 별로 역전파 계산 진행
            dout = layer.backward(dout)
        # 최종 역전파 계산 값 리턴
        return dout
