# PTB 데이터셋에 통계 기반 기법 적용

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

# 동시발생 행렬 계산 시에 설정할 윈도우 크기(주변 단어 범위)를 2로 설정
window_size = 2
# 단어 벡터 크기를 100으로 설정
wordvec_size = 100

# ptb 훈련용 데이터셋을 불러와 단어 ID 행렬, {단어 : ID} 딕셔너리, {ID : 단어} 딕셔너리를 저장
corpus, word_to_id, id_to_word = ptb.load_data('train')
# {단어 : ID} 딕셔너리의 원소 수를 측정하여 단어 개수 저장
vocab_size = len(word_to_id)
# 동시발생 수 계산 출력
print('동시발생 수 계산 ...')
# 단어 ID 행렬과 단어 크기, 윈도우 크기를 입력하여 동시발생 행렬을 계산하고 C에 저장
C = create_co_matrix(corpus, vocab_size, window_size)
# PPMI 계산 출력
print('PPMI 계산 ...')
# 동시발생 행렬을 입력하여 PPMI 계산 후 W에 저장(진행 상황 출력(verbose=True))
W = ppmi(C, verbose=True)

# 차원 감소를 위한 SVD 계산 출력
print('SVD 계산 ...')
# sklearn 모듈이 설치되어있다면 아래 내용 실행
try:
    # sklearn 모듈의 randomized_svd 함수 불러옴
    from sklearn.utils.extmath import randomized_svd
    # randomized_svd 함수를 통해 W 행렬을 U,S,V 3개 행렬로 분해(고속 SVD 계산법 사용), 매개변수로 축소할 차원 수를 미리 설정한 단어 벡터 크기(100)으로 하고, 반복 횟수를 5로 하며, 결과 재현 안 하는 것으로 설정
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
# sklearn 모듈이 없다면(import에 실패하여 ImportError 발생 시)
except ImportError:
    # 기존 numpy 모듈의 (느린) SVD 계산법으로 W 행렬을 U,S,V 3개 행렬로 분해
    U, S, V = np.linalg.svd(W)

# 단어 행렬의 밀집벡터 U에서 앞에서 설정한 단어 크기(100)만큼 수를 제한
word_vecs = U[:, :wordvec_size]

# 단어 쿼리 목록을 'you', 'year', 'car', 'toyota' 4가지로 설정
querys = ['you', 'year', 'car', 'toyota']
# 단어 쿼리 목록에서 각 단어 쿼리 별로 반복
for query in querys:
    # prac2-5에서 구현했던 가장 유사한 단어 목록을 나열해주는 함수(most_similar)를 통해 현재 단어 쿼리와 가장 유사한 상위 5개의 단어 출력
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)