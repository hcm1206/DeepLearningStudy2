# 2층 신경망을 실제로 학습시키는 코드를 구현

import sys, os
# 상위 디렉토리의 파일 접근을 위한 코드
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
# 매개변수 경신을 위한 코드 불러옴
from common.optimizer import SGD
# spiral 데이터셋 불러옴
from dataset import spiral
import matplotlib.pyplot as plt
# prac1-15 코드와 동일한 TowLayerNet 클래스 불러옴
from files.two_layer_net import TwoLayerNet

# ============================== 1. 초기 하이퍼파라미터 설정 ======================================
# 최대 에폭(학습 단위, 모든 데이터셋을 1바퀴 도는 횟수) 수를 300으로 설정
max_epoch = 300
# 배치 크기(한 번 학습에 사용하는 데이터 양)를 30으로 설정
batch_size = 30
# 은닉층 개수를 10으로 설정
hidden_size = 10
# 학습률을 1.0으로 설정
learning_rate = 1.0

# ============================== 2. 데이터 로드 및 신경망 모델과 옵티마이저(매개변수 갱신법) 생성 ======================================
# spiral 데이터셋에서 입력값과 입력값에 따른 정답값 불러옴
x, t = spiral.load_data()
# TwoLayerNet 클래스를 통해 입력층 2개, 은닉층 10개, 출력층 3개의 신경망 모델 불러옴
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
# 옵티마이저로 학습률 1.0의 SGD 저장
optimizer = SGD(lr=learning_rate)

# ============================== 3. 학습에 사용하기 위한 변수 선언 ======================================
# 데이터 크기를 데이터셋에 저장된 데이터 크기로 부터 불러옴
data_size = len(x)
# 최대 반복 횟수를 데이터 크기와 배치 크기의 몫으로 저장
max_iters = data_size // batch_size
# 손실 값 합계 저장할 변수 선언
total_loss = 0
# 손실 횟수를 저장할 변수 선언
loss_count = 0
# 모든 손실 값을 저장할 빈 리스트 선언
loss_list = []

# 최대 에폭 수만큼 에폭 별로 반복
for epoch in range(max_epoch):
    # ============================== 4. 데이터 랜덤 셔플 ======================================
    # 0~데이터 크기 까지의 정수가 무작위로 섞인 배열 저장
    idx = np.random.permutation(data_size)
    # 입력값 배열을 무작위로 섞인 인덱스 배열 대로 저장
    x = x[idx]
    # 정답 레이블 배열을 무작위로 섞인 인덱스 배열 대로 저장
    t = t[idx]
    
    # 최대 반복 횟수만큼 반복 횟수 별로 반복
    for iters in range(max_iters):
        # 배치 크기 개수만큼 입력값 데이터를 묶어서 batch_x 행렬로 저장
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        # 배치 크기 개수만큼 정답 레이블 데이터를 묶어서 batch_t 행렬로 저장
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # ============================== 5. 기울기를 구하여 매개변수 경신 ======================================
        # 신경망 모델에서 입력값 배치와 정답값 배치를 입력하여 순전파 계산 후 손실 값을 loss로 저장
        loss = model.forward(batch_x, batch_t)
        # 순전파 계산 이후 신경망 모델의 역전파 계산
        model.backward()
        # 옵티마이저에 저장된 SGD 기법으로 신경망 모델의 매개변수와 기울기 갱신
        optimizer.update(model.params, model.grads)

        # 손실 값 합계 변수에 현재 손실 값을 더하여 저장
        total_loss += loss
        # 손실 횟수에 1을 더하여 저장
        loss_count += 1

        # ============================== 6. 학습 경과를 정기적으로 출력 ======================================
        # 10번 반복시마다 아래 내용 실행
        if (iters+1) % 10 == 0:
            # 평균 손실값 계산하여 avg_loss로 저장
            avg_loss = total_loss / loss_count
            # 현재 에폭 수, 현재 반복 수, 현재 평균 손실값을 출력
            print('| 에폭 %d | 반복 %d / %d | 손실 %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
            # 전체 손실값 리스트에 현재 평균 손실값을 새 원소로 저장
            loss_list.append(avg_loss)
            # 전체 손실값, 손실 횟수를 0으로 초기화 (10번 반복할 동안의 손실값과 손실 횟수만 카운트 하고 10번 반복 후 출력이 된 후에는 초기화하는 것)
            total_loss, loss_count = 0, 0


# 실제 실행 시 