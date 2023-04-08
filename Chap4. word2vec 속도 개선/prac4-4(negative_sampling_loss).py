# 네거티브 샘플링 구현
# 네거티브 샘플링 과정과 샘플링 후의 손실 값을 계산하는 과정을 통합하여 NegativeSamplingLoss 계층으로 구현

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from files.negative_sampling_layer import EmbeddingDot, UnigramSampler
from common.layers import SigmoidWithLoss
import numpy as np

# 네거티브 샘플링과 확률 및 손실 계산을 통합한 NegativeSamplingLoss 계층 클래스 구현
class NegativeSamplingLoss:
    # 클래스 생성자 정의, 가중치 W, 말뭉치 단어 ID corpus, 제곱할 값 power(기본값 0.75), 샘플링 값의 수 sample_size(기본값 5)를 입력받음
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        # 샘플링 값의 수 sample_size를 맴버변수로 저장
        self.sample_size = sample_size
        # UnigramSampler 클래스 객체에 말뭉치 단어 ID corpus, 제곱할 값 power, 샘플링 값의 수 sample_size를 입력하여 맴버변수 sampler로 저장
        self.sampler = UnigramSampler(corpus, power, sample_size)
        # (샘플링 값의 수 + 1)만큼의 SigmoidWithLoss 객체를 생성하여 리스트에 저장하고 맴버변수 loss_layer로 저장 (0번째는 정답, 나머지는 오답)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        # (샘플링 값의 수 + 1)만큼의 EmbeddingDot 객체를 가중치 W를 입력하여 생성한 후 리스트에 저장하고 맴버변수 embed_dot_layer로 저장 (0번째는 정답, 나머지는 오답)
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        # 매개변수와 기울기를 저장할 빈 리스트 생성하여 각각 맴버변수 params, grads로 저장
        self.params = self.grads = [], []
        # 맴버변수 embed_dot_layers의 각 EmbeddingDot 계층들을 반복변수 layer에 각각 대입하며 반복
        for layer in self.embed_dot_layers:
            # 각 EmbeddingDot의 매개변수들을 클래스 맴버변수 params에 추가하여 저장
            self.params += layer.params
            # 각 EmbeddingDot의 기울기들을 클래스 맴버변수 grads에 추가하여 저장
            self.grads += layer.grads

    # 순전파 구현, 은닉층 h, 정답(긍정적 예)값 타깃 target 입력받음
    def forward(self, h, target):
        # target 행렬의 0번째 축 크기(행)을 통해 배치 크기를 알아내어 batch_size로 저장
        batch_size = target.shape[0]
        # sampler 클래스의 get_negative_sample 메소드를 통해 정답값 타깃 target의 오답(부정적 예) 샘플링하여 negative_sample에 저장
        negative_sample = self.sampler.get_negative_sample(target)

        # EmbeddingDot 계층들이 저장된 embed_dot_layers 중 정답 데이터(0번째 인덱스)의 순전파 계산(은닉층 h와 정답 타깃값 target 이용)한 결과를 score에 저장
        score = self.embed_dot_layers[0].forward(h, target)
        # 배치 크기(batch_size) 길이의 1로 이루어진 배열(결과가 정답(Yes)임을 나타내는 벡터)을 생성하여 correct_lable로 저장
        correct_label = np.ones(batch_size, dtype=np.int32)
        # SigmoidWithLoss 계층들이 저장된 loss_layers 중 정답 데이터(0번째 인덱스)의 순전파 계산(점수 score와 1(Yes)이 저장된 correct_label 배열 이용)한 결과를 loss에 저장
        loss = self.loss_layers[0].forward(score, correct_label)

        # 배치 크기(batch_size) 길이의 0으로 이루어진 배열(결과가 오답(No)임을 나타내는 벡터)을 생성하여 negative_label로 저장
        negative_label = np.zeros(batch_size, dtype=np.int32)
        # 샘플링한 값 크기(sample_size)만큼 반복
        for i in range(self.sample_size):
            # 오답 벡터가 저장된 negative_sample 중 i번째 샘플링 값을 negative_target값으로 저장
            negative_target = negative_sample[:,i]
            # embed_dot_layers 중 현재 샘플링 값에 해당(i+1번째 인덱스)하는 EmbeddingDot 계층에 은닉층 h, 현재 샘플링 값 벡터 negative_target을 입력하여 순전파 계산 실행한 결과(해당 오답 벡터와 은닉층의 행렬곱)를 score에 저장
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            # loss_layers 중 현재 샘플링 값에 해당(i+1번째 인덱스)하는 SigmoidWithLoss 계층에 점수 score, 오답을 나타내는 벡터 negative_label을 입력하여 순전파 계산 실행한 결과(점수와 실제 값(오답, 0)과의 손실)를 loss에 더하여 저장
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        # 손실 loss값 리턴
        return loss
    
    # 역전파 구현, 역전파 입력값 dout(기본값 1) 입력받음
    def backward(self, dout=1):
        # 역전파 점수를 저장할 dh를 선언하고 0으로 초기화
        dh = 0
        # 각 샘플링 데이터별 loss_layer(SigmoidWithLoss 계층)와 embed_dot_layer(EmbeddingDot 계층)을 각각 l0, l1로 하여 반복
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            # l0(SigmoidWithLoss 계층)의 역전파 계산에 dout을 입력한 결과를 dscore에 저장
            dscore = l0.backward(dout)
            # l1(EmbeddingDot 계층)의 역전파 계산에 dscore를 입력한 결과를 dh에 더하여 dh값 갱신
            dh += l1.backward(dscore)

        # 최종 dh값 리턴
        return dh
    
    # 은닉층의 뉴런은 순전파 시에 여러(샘플링 데이터 수 만큼) 개로 복사되었으므로 역전파에서는 복사된 여러 개의 기울기 값을 더함