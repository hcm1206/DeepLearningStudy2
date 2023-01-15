# 확률적경사하강법 (Gtochastic Gradient Descent, SGD) 구현

# 신경망 학습에서 신경망의 매개변수를 갱신하는 방법 중 가장 간단한 방법이 확률적경사하강법(SGD)
# 무작위로 선택된 데이터(미니배치)에 대한 기울기를 구하고 그 기울기 방향으로 일정한 거리만큼 갱신
# 가중치가 W일 때, SGD는 손실함수 기울기와 학습률(Learning Rate)의 곱을 가중치 W에서 감산하여 갱신

import numpy as np

# SGD 매개변수 갱신 클래스 구현
class SGD:
    # 클래스 생성자(학습률 lr 입력, 기본값 0.01)
    def __init__(self, lr=0.01):
        # 학습률 lr을 클래스 변수로 저장
        self.lr = lr

    # 가중치 매개변수 갱신 메서드(매개변수 params 행렬, 기울기 grads 행렬 입력)
    def update(self, params, grads):
        # 반복변수 i를 선언하여 0부터 매개변수 params 행렬의 행의 수 만큼 반복(미니배치로 묶인 각 데이터별로 반복한다는 뜻)
        for i in range(len(params)):
            # i번째 매개변수 행렬을 기존 값에서 학습률(lr)과 i번째 기울기 행렬을 곱한 값을 감산한 값으로 저장
            params[i] -= self.lr * grads[i]