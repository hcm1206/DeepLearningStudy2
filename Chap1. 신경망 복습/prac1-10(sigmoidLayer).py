# 활성화 함수 역할을 하는 시그모이드 함수가 담긴 시그모이드 계층을 클래스로 구현

import numpy as np

# 시그모이드 계층 클래스
class Sigmoid:
    # 클래스 생성자
    def __init__(self):
        # 계층 매개변수와 기울기를 저장할 빈 리스트 생성
        self.params, self.grads = [], []
        # 계층 출력값을 저장할 변수를 생성 (값은 저장 안 하고 선언만 함)
        self.out = None

    # 순전파 계산 (순전파 입력값 x 입력)
    def forward(self, x):
        # 입력값 x를 시그모이드 함수에 입력한 결과를 출력값으로 저장
        out = 1 / (1 + np.exp(-x))
        # 출력값 out을 클래스 변수로 저장
        self.out = out
        # 출력값 out 리턴
        return out

    # 역전파 계산 (역전파 입력값 dout 입력)
    def backward(self, dout):
        # 역전파 출력값 변수 dx에 역전파 입력값 dout의 시그모이드 함수 역전파 계산 결과를 저장
        dx = dout * (1.0 - self.out) * self.out
        # 역전파 출력값 변수 dx 리턴
        return dx
        