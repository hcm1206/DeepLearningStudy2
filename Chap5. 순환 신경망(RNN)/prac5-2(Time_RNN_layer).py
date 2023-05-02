# Time RNN 계층은 T개의 RNN 계층으로 이루어진 순환을 이루는 신경망 계층
# 5-1에서 구현한 RNN 계층을 이어붙여 구현
# RNN 계층의 은닉상태 h를 인스턴스 변수로 유지하여 RNN 계층에서 은닉상태를 다음 RNN 계층으로 인계받는데 사용

# stateful은 '상태가 있는'이라는 의미로, TimeRNN 계층이 은닉상태를 유지(순전파를 끊지 않고 전파)한다는 뜻
# True/False로 구분되는 bool 변수로, False라면 은닉 상태를 모든 원소가 0인 행렬 영행렬로 초기화하여 상태를 없앰

# TimeRNN 계층 구현
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.time_layers import RNN

# TimeRNN 클래스 정의
class TimeRNN:
    # 클래스 생성자 정의 (입력값 가중치 Wx, 은닉층 가중치 Wh, 편향 b, stateful 변수(기본값 False))
    def __init__(self, Wx, Wh, b, stateful=False):
        # 맴버변수 매개변수(params)로 입력층 가중치 Wx, 은닉층 가중치 Wh, 편향 b을 받아와 저장
        self.params = [Wx, Wh, b]
        # 맴버변수 기울기(grads)로 입력층 가중치, 은닉층 가중치, 편향과 동일한 형상의 0으로 채워진 행렬 저장
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        # 각 RNN 계층을 저장할 변수를 선언
        self.layers = None

        # 은닉 상태 h와 은닉 상태의 기울기 dh 선언
        self.h, self.dh = None, None
        # stateful 변수를 클래스 맴버변수로 저장
        self.stateful = stateful

    # 은닉상태를 설정하는 set_state() 메소드 정의 (은닉 상태 h 입력받음)
    def set_state(self, h):
        # 현재 은닉상태를 입력받은 은닉상태로 설정
        self.h = h
    
    # 은닉상태를 초기화하는 reset_state() 메소드 정의
    def reset_state(self):
        # 현재 은닉상태 초기화
        self.h = None

    # 순전파 계산 메소드 구현(입력값으로 xs 행렬 받음)
    def forward(self, xs):
        # 맴버변수로 저장된 매개변수(params)로부터 입력층 가중치, 은닉층 가중치, 편향을 받아와 각각 Wx, Wh, b로 저장
        Wx, Wh, b = self.params
        # 입력받은 xs 행렬의 형상을 통해 미니배치 크기 N, 시계열 데이터(RNN 계층) 크기 T, 입력 벡터 차원 수 D를 구함
        N, T, D = xs.shape
        # 입력층 기울기의 형상을 통해 입력 벡터 차원 수 D, 은닉층 크기 H를 구함
        D, H = Wx.shape

        # 각 계층을 저장할 맴버변수 layers를 선언하여 빈 리스트 저장
        self.layers = []
        # 긱 시계열마다의 은닉 상태를 저장할 hs에 N(미니배치 크기) × T(시계열 데이터 크기) × H(은닉층 크기) 형상의 빈 실수 행렬 저장
        hs = np.empty((N, T, H), dtype='f')

        # stateful(은닉 상태)이 False이거나 순전파가 처음 호출되었다면
        if not self.stateful or self.h is None:
            # 은닉 상태 h를 N(미니배치 크기) × H(은닉층 크기) 형상의 0으로 이루어진 실수 행렬로 초기화
            self.h = np.zeros((N, H), dtype='f')

        # 시계열 데이터(RNN 계층) 개수 만큼 반복하며 t번째 시계열 데이터 조작
        for t in range(T):
            # 현재 매개변수를 입력받는 RNN 계층 객체 생성하여 layer로 저장
            layer = RNN(*self.params)
            # 현재 RNN 계층에 입력값 행렬 xs에서의 t번째 시계열에 해당하는 행렬과 현재 은닉 상태 self.h를 입력하여 순전파 계산한 결과를 현재 은닉 상태 self.h로 저장하여 갱신
            self.h = layer.forward(xs[:, t, :], self.h)
            # hs 행렬에서의 t번째 시계열에 해당하는 행렬을 현재 은닉 상태 값으로 갱신
            hs[:, t, :] = self.h
            # 각 RNN 계층이 저장된 layers 리스트에 현재 RNN 계층을 추가하여 저장
            self.layers.append(layer)

        # 최종 계산된 은닉상태 행렬 hs 리턴
        return hs
    
    # 역전파 메소드 구현(은닉상태 역전파 입력값 dhs 입력받음)
    def backward(self, dhs):
        # 맴버변수 매개변수로부터 입력값 가중치, 은닉층 가중치, 편향을 받아와 각각 Wx, Wh, b로 저장
        Wx, Wh, b = self.params
        # 은닉상태 역전파 입력값 dhs의 형상으로부터 미니배치 크기 N, 시계열 데이터(RNN 계층) 개수 T, 은닉층 크기 H를 구함
        N, T, H = dhs.shape
        # 입력층 가중치로부터 입력차원 벡터 수 D, 은닉층 크기 H 입력받음
        D, H = Wx.shape

        # 입력값 행렬 기울기를 저장할 dxs에 N(미니배치 크기) × T(시계열 데이터 크기) × D(입력차원 벡터 수) 크기의 빈 실수 벡터 저장
        dxs = np.empty((N, T, D), dtype='f')
        # 은닉층 기울기를 저장할 dh를 선언하고 0으로 초기화
        dh = 0
        # 기울기를 저장할 3개 원소(각각 Wx, Wh, b) 리스트 grads를 생성하고 원소를 0으로 초기화
        grads = [0,0,0]
        # 각 시계열 데이터(RNN 계층)을 반대 방향으로 반복
        for t in reversed(range(T)):
            # 이번 t번째 RNN 계층을 layer로 저장
            layer = self.layers[t]
            # 이번 RNN 계층 layer의 역전파 계산을 이번 RNN 계층에 해당하는 은닉층 기울기와 은닉 상태 기울기의 합산 값을 입력하여 수행한 후 입력값 x의 기울기, 은닉 상태 기울기를 구함
            dx, dh = layer.backward(dhs[:,t,:] + dh)
            # 이번 RNN 계층에 해당하는 입력값 행렬 xs의 기울기 행렬에 x의 기울기 저장
            dxs[:,t,:] = dx
            # 현재 RNN 계층 layer의 기울기들을 반복
            for i, grad in enumerate(layer.grads):
                # 기울기 리스트의 i번째 원소로 이번 기울기의 값을 더하여 저장
                grads[i] += grad

        # 각 기울기(grads) 별로 인덱스 i, 기울기값 grads를 통해 반복
        for i, grad in enumerate(grads):
            # 맴버변수 기울기에서 i번째 시계열 기울기로 현재 기울기 저장
            self.grads[i][...] = grad
        # 은닉층 기울기 dh를 클래스 맴버변수로 저장
        self.dh = dh

        # 최종 계산된 입력값 행렬 기울기 dxs 리턴
        return dxs

