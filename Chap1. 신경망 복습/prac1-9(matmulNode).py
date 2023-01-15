# 행렬곱(Matmul, Matrix Multiply) 계층을 클래스로 구현

import numpy as np

# Matmul 계층(노드) 클래스
class MatMul:
    # 클래스 생성자 (입력값에 행렬곱 계산을 할 행렬 W 입력)
    def __init__(self,W):
        # Matmul 계층의 매개변수로 입력받은 W 행렬을 저장
        self.params = [W]
        # 매개변수의 기울기를 저장하기 위해 매개변수 W 행렬과 같은 형상의 0으로 이루어진 행렬 생성
        self.grads = [np.zeros_like(W)]
        # 입력값을 저장할 변수 x 선언 (값은 저장하지 않음)
        self.x = None

    # 순전파 계산 (순전파 입력값 x 입력받음)
    def forward(self,x):
        # 클래스의 매개변수를 불러와 W 행렬로 저장
        W, = self.params
        # 출력값 out을 입력값 행렬 x와 W 행렬의 행렬곱으로 계산
        out = np.matmul(x, W)
        # 입력값으로 입력된 x 행렬을 클래스에 저장
        self.x = x
        # 출력값 out 리턴
        return out

    # 역전파 계산 (역전파 입력값 dout 입력받음)
    def backward(self, dout):
        # 클래스의 매개변수를 불러와 W 행렬로 저장
        W, = self.params
        # 입력값 x의 역전파 출력값은 역전파 입력값 dout과 매개변수 행렬 W의 형상을 뒤집은 행렬(W.T)을 행렬곱 계산하여 구함
        dx = np.matmul(dout, W.T)
        # 매개변수 W의 역전파 출력값은 입력값 행렬 x의 형상을 뒤집은 행렬(x.T)과 역전파 입력값 dout을 행렬곱 계산하여 구함
        dW = np.matmul(self.x.T, dout)
        # 매개변수의 기울기(self.grads) 행렬의 0번째 차원 위치에 매개변수 W의 역전파 출력값 저장 (깊은 복사)
        self.grads[0][...] = dW
        # 입력값 x의 역전파 출력값 리턴
        return dx

    # 순전파 계산 시 
    # (N × D) 형상의 입력값 행렬 x와 (D × H) 형상의 매개변수 행렬 W을 행렬곱
    # 순전파 출력값 out은 (N × H) 형상의 행렬
    # 역전파 계산 시 
    # 역전파 입력값 dout은 (N × H) 형상의 행렬 (순전파 출력값 out 행렬 형상과 같음)
    # 역전파 출력값 dx는 (N × H) 형상의 역전파 입력값 dout 행렬과 매개변수 행렬 W의 형상을 뒤집은 (H × D) 형상의 W.T 행렬을 행렬곱
    # dx는 (N × D) 형상의 행렬 (순전파 입력값 x 행렬 형상과 같음)
    # 역전파 출력값 dW는 입력값 행렬 x의 형상을 뒤집은 (D × N) 행렬과 (N × H) 형상의 역전파 입력값 dout 행렬을 행렬곱
    # dW은 (D × H) 형상의 행렬 (매개변수 W 행렬 형상과 같음)