# 정규화 함수인 소프트맥스 함수와 손실 함수인 교차 엔트로피 오차 함수를 동시에 적용한 SoftmaxWithLoss 계층을 클래스로 구현

import numpy as np

# SoftmaxWithLoss 계층 클래스
class SoftmaxWithLoss:
    # 클래스 생성자
    def __init__(self):
        # 계층 매개변수, 기울기를 저장할 빈 리스트를 각각 생성
        self.params, self.grads = [], []
        # 소프트맥스 함수의 결과값을 저장할 변수 y를 클래스 변수로 선언 (선언만 하고 값은 저장 안 함)
        self.y = None
        # 결과값과 비교할 정답 데이터를 저장할 변수 t를 클래스 변수로 선언 (선언만 하고 값은 저장 안 함)
        self.t = None
    
    # 순전파 구현(입력값 행렬 x, 정답 데이터(원-핫 레이블) t 입력)
    def forward(self, x, t):
        # 원-핫 레이블로 구현된 정답 데이터 행렬을 클래스 변수로 저장
        self.t = t
        # 입력값 x를 소프트맥스 함수에 입력한 계산 결과를 클래스 변수 y로 저장
        self.y = softmax(x)
        # 소프트맥스 함수 결과 y 행렬과 정답 데이터 t 행렬의 원소 크기가 같다면(정답 레이블이 원-핫 벡터일 경우)
        if self.t.size == self.y.size:
            # 정답 데이터 t에 정답 데이터 t 행렬의 1번째 축에서 가장 큰 값의 인덱스 저장(정답 인덱스로 변환하는 작업)
            self.t = self.t.argmax(axis=1)
        # 소프트맥스 계산 결과 행렬 y와 정답 데이터 행렬 t를 교차 엔트로피 오차 함수에 입력하여 손실 값 loss 계산
        loss = cross_entrophy_error(self.y, self.t)
        # 손실 값 loss 리턴
        return loss
    
    # 역전파 구현(역전파 입력값 dout 입력, 기본값은 1)
    def backward(self, dout=1):
        # 정답 데이터의 0번째 차원 크기를 배치 크기(batch_size)로 저장
        batch_size = self.t.shape[0]
        # 역전파 출력값 dx에 소프트맥스 출력값 y와 같은 값 저장
        dx = self.y.copy()
        # 역전파 출력값 dx의 0번째 인덱스에 0~배치 크기 미만 값까지의 행렬, 1번째 인덱스에 정답 데이터 행렬에 1을 뺀 값 저장
        dx[np.arange(batch_size), self.t] -= 1
        # 역전파 출력값 dx에 역전파 입력값 dout을 곱하여 저장
        dx *= dout
        # 역전파 출력값 dx를 배치 크기(batch_size)로 나눈 값 저장
        dx = dx / batch_size

        # 역전파 출력값 dx 리턴
        return dx
        

# 소프트맥스 함수 구현
def softmax(x):
    # 입력값 행렬 x의 원소 중에서 가장 큰 원소를 c로 저장
    c = max(x)
    # 입력값 행렬의 원소들을 모두 c(행렬에서 가장 큰 원소)로 뺀 값을 자연상수 e의 지수로 하여 exp_x 행렬로 저장
    exp_x = np.exp(x-c)
    # exp_x 행렬의 모든 원소를 더하여 sum_exp_x에 저장
    sum_exp_x = np.sum(exp_x)
    # exp_x 행렬의 원소들을 sum_exp_x 값으로 각각 나눈 행렬을 y 행렬로 저장
    y = exp_x/sum_exp_x
    # y 행렬 리턴
    return y

# 교차 엔트로피 오차 함수 구현(입력 데이터 행렬 y, 정답 데이터 t 입력)
def cross_entrophy_error(y, t):
    # 오류 방지를 위한 아주 작은 값을 delta로 저장
    delta = 1e-7
    # 교차 엔트로피 오차 함수 계산 결과(y 행렬의 각 원소에 delta를 더한 후 로그를 취한 값을 정답 데이터 t와 곱한 행렬을 모두 더하고 음수를 붙인 값) 리턴
    return -np.sum(t * np.log(y+delta))