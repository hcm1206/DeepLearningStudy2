# word2vec 신경망에서의 입력값은 다수의 맥락(contexts)이고, 출력값은 타깃(target)
# 타깃은 말뭉치에서 찾는 목표가 되는 단어이고, 맥락은 타깃 단어의 일정 범위 주변 단어(들)을 의미
# 따라서 신경망 입출력 데이터를 처리하기 위해 말뭉치 텍스트를 prac2-2에서 진행한 바와 같이 ID로 변환하는 작업 필요

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# prac2-2에서 구현한 말뭉치 문자열에서 단어를 뽑아 인덱스화하는 전처리 함수 불러옴
from common.util import preprocess

# 처리할 말뭉치를 text에 저장
text = 'You say goodbye and I say hello.'
# 전처리함수(preprocess())에 말뭉치 text를 입력하여 단어의 ID 행렬(corpus), {단어 : ID} 쌍의 딕셔너리(word_to_id), {ID : 단어} 쌍의 딕셔너리(id_to_word)를 구함
corpus, word_to_id, id_to_word = preprocess(text)

# text 말뭉치의 단어 ID 행렬 출력
print(corpus)

# text 말뭉치의 {ID : 단어} 쌍의 딕셔너리 출력
print(id_to_word)

# 이렇게 생성된 말뭉치 인덱스 데이터를 통해 대응되는 각각의 맥락과 타깃을 지정하는 함수를 작성

import numpy as np

# 단어 ID 행렬(corpus)과 맥락 단어 범위(window_size ,기본 값 1)를 입력받아 맥락과 타깃 쌍을 구하는 함수 정의
def create_contexts_target(corpus, window_size=1):
    # 타깃 값을 단어 ID 행렬(corpus)에서 뽑아와 행렬로 지정(맥락 범위를 벗어나는 양 끝 부분 단어는 제외)
    target = corpus[window_size : -window_size]
    # 타깃 값에 대응되는 맥락 값을 저장할 빈 리스트 생성
    contexts = []

    # 반복변수 idx를 이용하여 단어 ID 행렬 중 맥락 범위를 벗어나는 양 끝 부분 단어를 제외한 단어들의 개수만큼 반복
    for idx in range(window_size, len(corpus)-window_size):
        # 현재 해당하는 인덱스(idx) 타깃값의 맥락값들을 저장할 빈 리스트 cs 생성
        cs = []
        # 반복 변수 t를 이용하여 앞뒤 맥락 단어 범위만큼 반복 (맥락 단어 범위가 1이면 -1부터 1까지 3번 반복)
        for t in range(-window_size, window_size + 1):
            # 현재 맥락 범위 t가 0이면
            if t == 0:
                # 타깃 값을 나타내는 것이므로 생략하고 다음 반복 진행
                continue
            # 현재 해당하는 조회 중인 인덱스의 맥락 단어를 cs 리스트에 원소로 추가
            cs.append(corpus[idx + t])
        # 현재 해당하는 인덱스(idx) 타깃값의 맥락값 리스트를 원소로 하여 전체 맥락 리스트 context에 원소로 추가
        contexts.append(cs)

    # 전체 맥락 리스트와 전체 타깃 리스트를 넘파이 배열로 변환하여 리턴
    return np.array(contexts), np.array(target)

# 위 함수가 실행되면 유효한 타깃값이 target 배열로 리턴되고, context 배열에는 각 타깃값의 맥락값들이 target 배열의 해당하는 타깃값와 같은 인덱스로 저장되어 있음

# 위에서 생성한 text 말뭉치의 단어 ID 배열을 맥락 범위 1로 하여 create_contexts_target() 함수를 통해 맥락값(contexts), 타깃값(target)을 구함
contexts, target = create_contexts_target(corpus, window_size=1)
# 생성된 맥락값(contexts) 출력
print(contexts)
# 생성된 타깃값(target) 출력
print(target)

# target 배열의 0번 인덱스에는 타깃값 중 하나인 say 단어의 ID(1)가 저장되어 있고
# contexts 배열의 0번 인덱스에는 say 단어의 맥락값에 해당되는 you(0), hello(2)의 ID가 저장되어 있음
# 이처럼 create_contexts_target() 함수로 생성한 contexts, target 배열들은 각각의 인덱스에 대응

print()

# 위에서 생성된 contexts와 target 배열을 CBOW 모델에서 사용하기 위해 해당 배열들을 원-핫 벡터 표현으로 바꾸어야 함

# 단어 ID 행렬(corpus)과 어휘 숫자(vocab_size)를 입력받아 원-핫 벡터 표현으로 변환하는 함수 convert_ont_hot() 정의
def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환

    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    # 단어 ID 행렬의 열의 수(입력받은 처리할 단어 개수)를 N에 저장
    N = corpus.shape[0]

    # 단어 ID 행렬의 차원 수가 1이면(ex. 각 인덱스에 해당하는 단어가 하나 밖에 없는 '타깃값'이면)
    if corpus.ndim == 1:
        # (처리할 데이터 개수(N) × 어휘 개수(vocab_size)) 형상의 32비트 정수를 저장하는 모든 원소가 0으로 이루어진 행렬 생성하여 one_hot으로 저장
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        # 단어 ID 행렬의 원소들을 반복하며 각 인덱스를 idx, 단어 ID를 word_id로 지정
        for idx, word_id in enumerate(corpus):
            # one_hot 벡터의 현재 인덱스(idx)의 단어(타깃값)에 해당하는 단어 ID(word_id)를 가리키는 원소에 1 저장
            one_hot[idx, word_id] = 1

    # 단어 ID 행렬의 차원 수가 2이면(ex. 각 인덱스에 해당하는 단어가 2개 이상인 '맥락값들'이면)
    elif corpus.ndim == 2:
        # 단어 ID 행렬의 행의 수(타깃값이 갖는 맥락 값의 수, window_size*2)를 구하여 C에 저장
        C = corpus.shape[1]
        # (처리할 데이터의 수(N) × 하나의 타깃값이 갖는 맥락값의 수(C) × 어휘 개수(vocab_size)) 형상의 32비트 정수를 저장하는 모든 원소가 0으로 이루어진 행렬 생성하여 one_hot으로 저장
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        # 단어 ID 행렬의 원소들을 반복하며 각 인덱스를 idx_0, 단어 ID 2차원 행렬을 word_ids로 지정
        for idx_0, word_ids in enumerate(corpus):
            # 단어 2차원 행렬 word_ids의 원소들을 반복하며 각 인덱스를 idx_1, 단어 ID 배열을 word_id로 지정
            for idx_1, word_id in enumerate(word_ids):
                # one_hot 벡터의 현재 인덱스(idx_0)의 단어(타깃값)에 해당하는 맥락 단어(idx_1)의 단어 ID(word_id)를 가리키는 원소에 1 저장
                one_hot[idx_0, idx_1, word_id] = 1

    # 생성된 one_hot 행렬 리턴
    return one_hot

# 위 함수를 실행하면 입력한 단어 ID 목록을 원-핫 형태로 변환한 행렬값이 리턴됨

print()

# {단어 : ID} 쌍의 딕셔너리를 이용하여 어휘 개수를 구해 vocab_size로 저장
vocab_size = len(word_to_id)
# convert_one_hot() 함수를 이용하여 target 행렬의 타깃값을 원-핫 행렬로 변환하여 저장
target = convert_one_hot(target, vocab_size)
# convert_one_hot() 함수를 이용하여 contexts 행렬의 맥락값을 원-핫 행렬로 변환하여 저장
contexts = convert_one_hot(contexts, vocab_size)

# 원-핫 벡터로 변환된 타깃값(target)과 맥락값(contexts) 출력
print(target)
print(contexts)