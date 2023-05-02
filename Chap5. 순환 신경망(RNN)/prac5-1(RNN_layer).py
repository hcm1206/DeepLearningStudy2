# RNN(순환 신경망)은 데이터가 순환하는 모양의 신경망으로, 과거의 데이터가 현재의 데이터 출력에 영향을 주고 현재 데이터의 출력이 미래 데이터에 영향을 줌
# RNN은 데이터가 여러 개의 동일한 계층을 순환하는 방식으로 구현

# 순환하며 거치는 하나의 RNN 계층을 구현
import numpy as np

# RNN 클래스 정의
class RNN:
    # 생성자 정의(입력층 가중치, 은닉층 가중치, 편향을 입력받음)
    def __init__(self, Wx, Wh, b):
        # 맴버변수 매개변수(params)로 입력층 가중치, 은닉층 가중치, 편향 저장
        self.params = [Wx, Wh, b]
        # 맴버변수 기울기(grads)로 입력층 가중치, 은닉층 가중치, 편향과 동일한 형상의 0으로 채워진 행렬 저장
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]

    # 순전파 계산(입력값 x, 이전 RNN 계층의 은닉층 입력받음)
    def forward(self, x, h_prev):
        # 맴버변수 매개변수(params)로부터 입력층 가중치, 은닉층 가중치, 편향을 받아와 각각 Wx, Wh, b로 저장
        Wx, Wh, b = self.params
        # 매개변수 행렬곱(이전 RNN 은닉층 × 은닉층 가중치, 입력층 × 입력층 가중치)과 편향 덧셈을 통해 은닉층 계산
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        # 계산된 은닉층에 쌍곡탄젠트(tanh) 계산하여 다음 RNN 계층으로 보낼 h_next로 저장
        h_next = np.tanh(t)

        # 객체 임시변수 cache로 입력값 x, 이전 계층에서 받아온 은닉층 h_prev, 다음 계층으로 보낼 은닉층 h_next 저장
        self.cache = (x, h_prev, h_next)
        # 다음 계층으로 전달하기 위해 h_next 리턴
        return h_next

    # 역전파 계산(다음 은닉층에서 받아오는 역전파 계산값 dh_next 입력받음)
    def backward(self, dh_next):
        # 맴버변수 매개변수(params)로부터 입력층 가중치, 은닉층 가중치, 편향을 받아와 각각 Wx, Wh, b로 저장
        Wx, Wh, b = self.params
        # 맴버변수 cache로부터 입력값, 이전 계층에서 받은 은닉층, 다음 계층으로 보낸 은닉층을 받아와 각각 x, h_prev, h_next로 저장
        x, h_prev, h_next = self.cache

        # 은닉층 계산값 t의 역전파 계산 (tanh 역전파 계산 dx = dy*(1-y**2))
        dt = dh_next * (1 - h_next ** 2)
        # 편향 b의 역전파 계산 (덧셈 역전파 계산 dx = dt 행렬의 0축에 대한 합)
        db = np.sum(dt, axis=0)
        # 은닉층 가중치 Wh의 역전파 계산 (행렬곱 역전파 계산 dW = x.T × dy)
        dWh = np.matmul(h_prev.T, dt)
        # 이전 은닉층 h_prev의 역전파 계산 (행렬곱 역전파 계산 dx = dy × W.T)
        dh_prev = np.matmul(dt, Wh.T)
        # 입력값 가중치 Wx의 역전파 계산 (행렬곱 역전파 계산 dW = x.T × dy)
        dWx = np.matmul(x.T, dt)
        # 입력값 x의 역전파 계산 (행렬곱 역전파 계산 dx = dy × W.T)
        dx = np.matmul(dt, Wx.T)

        # 맴버변수 기울기(grads)의 0번째 행렬에 입력값 가중치 기울기 dWx를 행렬로 저장
        self.grads[0][...] = dWx
        # 맴버변수 기울기(grads)의 1번째 행렬에 은닉층 가중치 기울기 dWh를 행렬로 저장
        self.grads[1][...] = dWh
        # 맴버변수 기울기(grads)의 2번째 행렬에 편향 기울기 db를 행렬로 저장
        self.grads[2][...] = db

        # 입력값 기울기 dx, 이전 은닉층 기울기 dh_prev를 이전 RNN 계층 역전파 입력값으로 보내기 위해 리턴
        return dx, dh_prev
    
# 여기서 구현한 RNN 클래스는 common/time_layers.py의 RNN 클래스와 동일