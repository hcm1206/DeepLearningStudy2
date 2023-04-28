# 구현한 CBOW 모델을 PTB 데이터셋을 이용하여 훈련하는 코드 구현

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common import config

# 아래 코드는 GPU를 통해 쿠파이를 사용하기 위한 코드로, GPU와 쿠파이가 없다면 주석처리
# =======================================
# config.GPU = True
# =======================================

import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from files.cbow import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

# 맥락 윈도우 크기를 5로 설정
window_size = 5
# 은닉층 개수를 100으로 설정
hidden_size = 100
# 배치 크기를 100으로 설정
batch_size = 100
# 최대 에폭 수를 10으로 설정
max_epoch = 10

# PTB 데이터셋에서 훈련용 데이터를 받아와 말뭉치 ID(corpus), {단어 : ID} 딕셔너리(word_to_id), {ID : 단어} 딕셔너리(id_to_word)로 저장
corpus, word_to_id, id_to_word = ptb.load_data('train')
# {단어 : ID} 딕셔너리의 원소 크기를 통해 어휘 개수를 구하여 vocab_size로 저장
vocab_size = len(word_to_id)

# cretae_contexts_target() 함수를 통해 말뭉치 ID(corpus)와 맥락 윈도우 크기(window_size)에 맞는 문맥(context) 단어와 타깃(target) 단어 생성
contexts, target = create_contexts_target(corpus, window_size)
# 쿠파이 사용 코드
if config.GPU:
    contexts, target = to_gpu(contexts), to_cpu(target)

# 어휘 개수, 은닉층 개수, 맥락 윈도우 크기, 말뭉치 ID를 통해 CBOW 모델을 생성하고 model로 저장
model = CBOW(vocab_size, hidden_size, window_size, corpus)
# 최적화 기법으로 Adam 기법 객체 사용
optimizer = Adam()
# 학습을 위한 Trainer 객체 trainer를 생성하고 CBOW 모델(model)과 Adam 최적화 기법(optimizer) 입력
trainer = Trainer(model, optimizer)

# trainer 객체에 맥락(contexts), 타깃(target), 최대 에폭 수(max_epoch), 배치 크기(batch) 설정하여 학습 진행
trainer.fit(contexts, target, max_epoch, batch_size)
# 학습이 완료된 trainer 객체의 학습 과정 시각화
trainer.plot()

# 학습된 CBOW 모델에서 생성한 단어 분산 표현 벡터를 변수 word_vecs로 저장
word_vecs = model.word_vecs
# 쿠파이 사용 코드
if config.GPU:
    word_vecs = to_cpu(word_vecs)
# 매개변수를 저장할 빈 딕셔너리 생성
params = {}
# params 딕셔너리의 'word_vecs' 키값의 원소로 단어 분산 표현을 16비트 실수로 저장
params['word_vecs'] = word_vecs.astype(np.float16)
# params 딕셔너리의 'word_to_id' 키값의 원소로 {단어 : ID} 딕셔너리 저장
params['word_to_id'] = word_to_id
# params 딕셔너리의 'id_to_word' 키값의 원소로 {ID : 단어} 딕셔너리 저장
params['id_to_word'] = id_to_word
# 'cbow_params.pkl' 파일명으로 저장할 피클 데이터 pkl_file 생성
pkl_file = 'cbow_params.pkl'
# 피클 파일 pkl_file에 params 매개변수를 저장하여 파일로 추출
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)