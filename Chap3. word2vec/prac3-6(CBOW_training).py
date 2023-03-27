# 일반적인 신경망 학습과 같이 CBOW 모델의 학습을 구현
# 학습 데이터를 준비하여 신경망에 입력한 후 역전파 계산을 통해 기울기를 구하여 매개변수 갱신
# 1장에서 설명한 Trainer 클래스를 이용하여 학습 관련 부분을 Trainer 클래스에 위임

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.trainer import Trainer
from common.optimizer import Adam
# prac3-5에서 구현한 CBOW 모델 코드와 동일한 코드 불러옴
from files.simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

# ===================== 1. 기본 세팅 =====================

# 단어 맥락 범위를 1로 설정 (맥락 단어의 수 2개)
window_size = 1
# 은닉층 개수를 5로 설정
hidden_size = 5
# 배치 크기(1회 학습시 학습하는 데이터 수)를 3로 설정
batch_size = 3
# 최대 에폭(학습 횟수)수를 1000로 설정
max_epoch = 1000

# ===================== 2. 데이터 전처리 ===================== 

# 처리할 텍스트 데이터 문자열을 text에 저장
text = 'You say goodbye and I say hello.'
# 전처리 함수 preprocess()에 텍스트 문자열 text를 입력하여 단어 ID 행렬, {단어 : ID} 쌍 딕녀서리, {ID : 단어} 쌍 딕셔너리 구함
corpus, word_to_id, id_to_word = preprocess(text)

# {단어 : ID} 딕셔너리의 원소 수를 구하여 어휘 수(vacab_size)를 구함
vocab_size = len(word_to_id)
# 단어 ID 행렬과 지정된 맥락 단어 범위를 이용하여 create_contexts_target() 함수를 통해 맥락 단어와 타깃 단어를 구함
contexts, target = create_contexts_target(corpus, window_size)
# 타깃 단어를 원-핫 벡터 형태로 변환
target = convert_one_hot(target, vocab_size)
# 맥락 단어를 원-핫 벡터 형태로 변환
contexts = convert_one_hot(contexts, vocab_size)

# ===================== 3. 신경망 세팅 ===================== 

# pra3-5에서 구현했던 SimpleCBOW 클래스에 어휘 크기, 은닉층 개수를 입력하여 CBOW 신경망 모델을 생성하고 model로 저장
model = SimpleCBOW(vocab_size, hidden_size)
# Adam() 클래스 객체를 매개변수 갱신 기법인 optimizer로 저장
optimizer = Adam()
# Trainer 클래스에 model(SimpleCBOW 신경망 모델), optimizer(Adam 매개변수 갱신 기법)을 입력하여 trainer 객체 저장
trainer = Trainer(model, optimizer)

# ===================== 4. 신경망 학습 ===================== 

# trainer 객체 fit() 메소드에 맥락 벡터, 타깃 벡터, 최대 에폭 수, 배치 크기를 입력하여 신경망 학습 진행
trainer.fit(contexts, target, max_epoch, batch_size)
# 신경망 학습이 진행되는 과정의 손실값들을 그래프로 출력
trainer.plot()


# ===================== 5. 학습 정보 확인 ===================== 
# CBOW 신경망 모델(model)의 단어 밀집벡터(word_vecs)를 가져와 word_vecs로 저장
word_vecs = model.word_vecs
# {ID : 단어} 딕셔너리에서 각 쌍의 ID와 단어를 불러와 반복
for word_id, word in id_to_word.items():
    # 단어와 해당되는 단어의 단어 벡터를 출력
    print(word, word_vecs[word_id])

# 위에서 출력하는 정보는 각 단어마다 다른 단어와의 유사도가 저장된 단어 밀집벡터가 각각 출력됨
# 이번에 사용한 말뭉치는 매우 단순하고 작은 크기의 말뭉치이므로 좋은 결과는 아니지만 실용적이고 큰 말뭉치를 이용하면 좋은 결과 습득 가능
# 그러나 실용적인 말뭉치를 다루기 위해서는 성능을 향상시켜 더욱 효율적인 모델을 구현해야 함
# 4장에서 이번에 구현된 단순한 CBOW 모델을 개선한 실용적인 CBOW 모델 구현