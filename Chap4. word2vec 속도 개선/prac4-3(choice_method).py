# 앞서 이뤄진 이진분류에서는 정답에 관해서만 학습이 진행되고, 오답을 다룰 때는 학습이 되지 않음
# 시그모이드 계층에서 정답 입력 시 출력은 1에 가까워지고, 오답 입력 시 출력은 0에 가까워지도록 설정

# 그러나 모든 오답에 대하여 이진 분류를 학습시키는 것은 매우 비효율적(어휘 수 증가에 대처하기 위해 도입한 이진 분류가 의미가 없어짐)
# 따라서 부정적인 예시(오답) 몇 가지를 샘플로 뽑아(샘플링) 이러한 오답 입력값들을 학습하는 것이 네거티브 샘플링 기법

# 이 네거티브 샘플링에서 샘플링할 오답들은 무작위로 뽑는 것보다 출현 횟수에 따라 뽑는 것이 좋음
# 말뭉치에서 자주 등장하는 단어가 실제 선택될 확률이 높기 때문에 희소한 단어는 뽑힐 확률이 낮기 때문

# 확률 분포에 따른 샘플링 예시 : np.random.choice() 메소드 활용

import numpy as np

# 0~9까지 숫자 중 무작위로 하나 샘플링하여 출력
print(np.random.choice(10))
# 0~9까지 숫자 중 무작위로 하나 샘플링하여 출력
print(np.random.choice(10))

# words 말뭉치 생성
words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
# words 말뭉치 중 무작위로 하나 샘플링하여 출력
print(np.random.choice(words))

# words 말뭉치 중 무작위로 5개 샘플링하여 출력 (중복 포함)
print(np.random.choice(words, size=5))
# words 말뭉치 중 무작위로 5개 샘플링하여 출력 (중복 제거)
print(np.random.choice(words, size=5, replace=False))

# 총합이 1인 확률분포 원소가 포함된 리스트를 생성하여 p로 저장
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
# words 말뭉치를 확률분포 p에 따라 각 단어를 확률적으로 샘플링하여 하나 출력
np.random.choice(words, p=p)

print()

# word2vec의 네거티브 샘플링에서는 확률분포의 각 확률값들에 0.75를 제곱하라고 권고
# 이 과정을 거친 후 i번째 단어가 나타날 확률은 
# 기존에 i번째 단어가 나타날 확률에 0.75를 제곱한 값을 말뭉치의 각 단어들이 나타날 확률에 0.75를 제곱한 값들을 모두 더한 값으로 나눈 값
# 분모가 필요한 이유는 수정 후에도 확률의 총합이 1이 되어야 하기 때문

# 이 과정을 거치면 출현 확률이 낮은 단어가 버려지지 않고 확률을 살짝 높여 원래 확률보다 등장 확률이 다소 높아지도록 하는 효과를 가짐

# 위 과정에 대한 예시 코드

# 기존 확률 분포 리스트 p
p = [0.7, 0.29, 0.01]
# 기존 확률 분포 리스트 p 출력
print(p)
# 기존 확률분포 p의 원소들을 각각 0.75로 제곱한 새로운 확률분포 리스트를 new_p로 저장
new_p = np.power(p, 0.75)
# new_p의 각각의 원소들을 new_p 원소(새로운 확률분포)들의 총합으로 나누어 새 확률분포 리스트 new_p 완성
new_p /= np.sum(new_p)
# 완성된 새 확률분포 리스트 new_p 출력
print(new_p)

# 결과를 보면 기존에 낮은 확률 분포를 가지는 값들의 출현 확률이 새 확률분포에서 다소 높아진 것을 확인
# 이 과정을 통해 너무 낮은 출현 확률을 가진 값들의 확률을 다소 높여서 아예 배제되지 않고 비교적 균등하게 샘플링될 수 있도록 조치

print()

# 이 처리를 담당할 클래스는 부가적인 기능을 더하여 UnigramSampler라는 이름으로 제공
# 하나의 단어를 대상으로 확률 분포를 만드는 클래스
# UnigramSampler는 초기화 시 3개의 인수 입력(단어 ID 목록 corpus, 제곱값 power(기본값 0.75), 네거티브 샘플링 수행 횟수 sample_size)
# UnigramSampler 클래스에서는 get_negative_sample(target) 메소드 제공
# target 인자에 입력된 값 ID를 정답으로 인지하고 그를 제외한(오답) 단어 ID를 샘플링하여 제공

# files/negative_sampling_layer.py 파일에서 UnigramSampler 클래스를 불러와 사용 가능

# 사용 예시

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# UnigramSampler 클래스 불러옴
from files.negative_sampling_layer import UnigramSampler

# 말뭉치 예시 배열 corpus 생성
corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
# 제곱할 값 power를 0.75로 설정
power = 0.75
# 샘플링할 값의 수를 2로 설정
sample_size = 2

# UnigramSampler 클래스에 말뭉치 corpus, 제곱할 값 power, 샘플링 값의 수 sample_size를 입력하여 sampler로 객체 생성
sampler = UnigramSampler(corpus, power, sample_size)
# 입력할 3개(미니배치)의 타깃값 배열을 생성하여 target으로 저장
target = np.array([1,3,0])
# sampler 객체의 get_negative_sample() 메소드에 target 행렬을 입력하여 뽑은 오답 샘플 값들을 negative_sampler에 저장
negative_sample = sampler.get_negative_sample(target)
# 뽑은 오답 샘플 negetive_sample 값 출력
print(negative_sample)

# 실행 결과는 말뭉치 중 각각의 정답 값(1,3,0)을 제외한 확률분포에 따른 랜덤 오답값 행렬