# 입력값(이전값)과 가중치를 행렬곱한 후 그 결과를 편향과 더하는 Affine 계층 구현
# 논리적으로는 Matmul 계층에서 편향 b를 추가한 계층

# 1. 입력값 X(N×D) 행렬과 가중치 W(D×H) 행렬을 행렬곱(MatMul)
# 2. 편향값 b(H)를 행렬곱(Matmul) 결과 y(N×D)의 0차원 크기 N만큼 반복(Repeat)
# 3. 반복(Repeat)된 편향값 b(N×H)와 행렬곱(MatMul) y(N×H)를 각 원소별 덧셈(sum)
# 4. 덧셈 결과가 최종 순전파 계산 결과 Z(N×H) 행렬

import numpy as np

# Affine 계층 클래스 구현
class Affine:
    # 클래스 생성자 (가중치 W, 편향 b 입력)
    def __init__(self, W, b):
        # Affine 계층 매개변수 리스트를 생성하여 0번째 인덱스에 가중치 W, 1번째 인덱스에 편향 b 저장
        self.params = [W, b]
        # Affine 계층 기울기 리스트를 생성하여 0번째 인덱스에 가중치 W와 동일한 형상의 0으로 이루어진 행렬, 1번째 인덱스에 편향 b와 동일한 형상의 0으로 이루어진 행렬 저장
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        # 순전파 입력값을 클래스 변수로 선언 (값은 저장하지 않고 선언만 함)
        self.x = None

    # 순전파 계산 (순전파 입력값 x 입력)
    def forward(self, x):
        # 클래스에 저장된 매개변수로부터 가중치와 편향을 가져와 W, b에 저장
        W, b = self.params
        # 입력값 x와 가중치 W를 행렬곱한 후 편향을 더한 값을 출력값 out으로 저장
        out = np.matmul(x, W) + b
        # 출력값 out을 클래스 변수로 저장
        self.out = out
        # 출력값 out 리턴
        return out

    # 역전파 계산 (역전파 입력값 dout 입력)
    def backward(self, dout):
        # 클래스에 저장된 매개변수로부터 가중치와 편향을 가져와 W, b에 저장
        W, b = self.params
        # 역전파 입력값 dout과 가중치 매개변수의 형상을 뒤집은 행렬을 행렬곱한 값을 역전파 출력값 dx로 저장
        dx = np.matmul(dout, W.T)
        # 순전파 입력값 x의 형상을 뒤집은 행렬과 역전파 입력값 dout을 행렬곱한 값을 역전파 출력값 dW로 저장
        dW = np.matmul(self.x.T, dout)
        # 역전파 입력값 dout의 행렬을 0축 방향으로 모두 더한 결과값을 역전파 출력값 db로 저장
        db = np.sum(dout, axis=0)

        # Affine 계층 기울기 리스트의 0번째 인덱스인 가중치 W 행렬에 dW의 값을 깊은 복사하여 저장(해당 메모리 위치에 dW 저장)
        self.grads[0][...] = dW
        # Affine 계층 기울기 리스트의 1번째 인덱스인 편향 b 행렬에 db의 값을 깊은 복사하여 저장(해당 메모리 위치에 db 저장)
        self.grads[1][...] = db
        # 역전파 출력값 dx 리턴
        return dx