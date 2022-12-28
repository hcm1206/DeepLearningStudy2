# 신경망에서의 추론 과정을 클래스를 이용환 모듈화로 구현
# 추론은 순전파를 통해 이루어지며, 지금은 역전파를 이용한 신경망 학습은 구현하지 않음

import numpy as np

# 시그모이드 함수를 이용한 활성화 계층인 Sigmoid 계층을 클래스로 구현
class Sigmoid:
    # 클래스 생성자
    def __init__(self):
        # Sigmoid 계층에서 사용되는 매개변수를 저장할 리스트 선언(현재는 없음)
        self.params = []

    # 클래스 순전파 구현 (입력값 x를 받음)
    def forward(self, x):
        # 입력값 x를 시그모이드 함수에 입력한 결과값을 리턴
        return 1 / (1 + np.exp(-x))

# 가중치와 편향을 적용하여 계산하는 계층인 Affine 계층을 클래스로 구현
class Affine:
    # 클래스 생성자 (가중치 W와 편향 b를 클래스 생성 시 입력받음)
    def __init__(self, W, b):
        # Affine 계층에서 사용되는 매개변수를 저장할 리스트 선언 (가중치 W, 편향 b 저장)
        self.params = [W, b]

    # 클래스 순전파 구현 (입력값 x를 받음)
    def forward(self, x):
        # 매개변수 리스트에서 가중치 W, 편향 b를 받아옴
        W, b = self.params
        # 입력값 x와 가중치 W를 행렬곱한 후 편향 b를 각각 더하여 결과값 저장
        out = np.matmul(x, W) + b
        # 결과값 리턴
        return out

# 입력층, 은닉층, 출력층으로 구성된 2번의 가중치 계산을 하는 2층 신경망 클래스 구현 (순전파만 구현)
class TwoLayerNet:
    # 클래스 생성자 (입력값 크기(입력 노드 개수), 은닉층 크기(은닉층 노드 개수), 출력값 크기(출력 노드 개수)를 클래스 생성 시 입력받음)
    def __init__(self, input_size, hidden_size, output_size):
        # 입력값 크기를 I, 은닉층 크기를 H, 출력층 크기를 O로 저장
        I, H, O = input_size, hidden_size, output_size

        # -1 ~ 1 사이의 랜덤 실수 값이 저장된 (입력값 크기 × 은닉층 크기) 형상의 첫번째 가중치 행렬 W1 생성
        W1 = np.random.randn(I, H)
        # -1 ~ 1 사이의 랜덤 실수 값이 저장된 (1 × 은닉층 크기) 형상의 첫번째 편향 행렬 b1 생성
        b1 = np.random.randn(H)
        # -1 ~ 1 사이의 랜덤 실수 값이 저장된 (은닉층 크기 × 출력층 크기) 형상의 두번째 가중치 행렬 W2 생성
        W2 = np.random.randn(H, O)
        # ~1 ~ 1 사이의 랜덤 실수 값이 저장된 (1 × 은닉층 크기) 형상의 두번쨰 편향 행렬 b2 생성
        b2 = np.random.randn(O)

        # 신경망의 각 계층을 리스트로 저장 
        self.layers = [
            # W1, b1 매개변수가 저장된 Affine(1) 계층
            Affine(W1, b1),
            # 활성화함수 Sigmoid 계층
            Sigmoid(),
            # W2, b2 매개변수가 저장된 Affine(2) 계층
            Affine(W2, b2)
        ]

        # 신걍망에서 사용되는 모든 매개변수를 저장할 리스트 생성
        self.params = []
        # 신경망의 각 계층 별로 반복 (Affine(1)->Sigmoid->Affine(2) 순으로 반복)
        for layer in self.layers:
            # 신경망 매개변수 리스트에 각 계층의 매개변수들을 추가
            self.params += layer.params

    # 신경망 추론(순전파) 메서드 (입력값 x를 입력받음)
    def predict(self, x):
        # 신경망의 각 계층 별로 반복 (Affine(1)->Sigmoid->Affine(2) 순으로 반복)
        for layer in self.layers:
            # x에 현재 x값을 각 계층의 forward() 메서드에 입력한 순전파 계산 결과 값으로 새로 저장
            x = layer.forward(x)
        # 최종 x 값 리턴
        return x

# x에 10 × 2 형상의 입력값 행렬 저장 (2 크기의 데이터 10개를 입력값으로 사용)
x = np.random.randn(10, 2)
# 신경망 모델을 2 크기의 입력층, 4 크기의 은닉층, 3 크기의 출력층으로 구성하여 model에 저장
model = TwoLayerNet(2, 4, 3)
# model에 저장된 신경망 모델에서 입력값 x를 순전파 계산한 결과를 s에 저장
s = model.predict(x)