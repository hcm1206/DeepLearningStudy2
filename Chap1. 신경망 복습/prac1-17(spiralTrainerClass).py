# 책에서 제공하는 Trainer 클래스 사용 테스트
# prac1-16 코드와 유사하나 몇 가지 기능이 추가되어 있는 Trainer 클래스를 통해 코드 간소화 및 추가 기능 사용 가능

import sys, os
# 상위 디렉토리의 파일 접근을 위한 코드
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# 책에서 제공하는 매개변수 갱신용 SGD 기법 클래스 불러옴
from common.optimizer import SGD
# 책에서 제공하는 신경망 학습용 Trainer 클래스 불러옴
from common.trainer import Trainer
# spiral 데이터셋 불러옴
from dataset import spiral
# prac1-15의 코드와 동일한 2층 신경망 불러옴
from files.two_layer_net import TwoLayerNet

# 최대 에폭을 300으로 설정
max_epoch = 300
# 배치 크기 30으로 설정
batch_size = 30
# 은닉층 10으로 설정
hidden_size = 10
# 학습률을 1.0으로 설정
learning_rate = 1.0

# spiral 데이터셋에서 입력값 x와 정답 레이블 t 불러옴
x, t = spiral.load_data()
# TwoLayerNet 클래스에서 입력층 2개, 은닉층 10개, 출력층 3개로 2층 신경망 모델 생성
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
# 옵티마이저로 학습률 1.0의 SGD 기법 생성
optimizer = SGD(lr=learning_rate)

# Trainer 클래스에 신경망 모델, 옵티마이저 입력하여 생성
trainer = Trainer(model, optimizer)
# Trainer 클래스에서 입력값 x, 정답 레이블 t, 최대 에폭 300, 배치 크기 30, 출력 시 반복 간격 10으로 하여 학습 진행
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
# Trainer 클래스를 통해 학습 정보가 담긴 그래프 출력
trainer.plot()

# 기본적인 기능은 prac1-16에서 구현한 신경망 학습 코드와 유사
# 그러나 신경망 학습 부분을 Trainer 클래스를 별도로 구현하여 담당하도록 변경 : 코드가 간결해짐
# 또한 반복 간격 간의 시간(밀리초 단위)을 평균손실 출력 시 같이 출력하고 평균 손실을 그래프에 출력하는 등 추가 기능 존재
# 앞으로 책에서 신경망 학습을 할 때 위와 같이 Trainer 클래스 사용 예정
