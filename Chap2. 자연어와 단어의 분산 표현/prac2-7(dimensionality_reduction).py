# 차원 감소(diemntionality reduction) : 중요한 정보는 유지하면서 벡터의 차원을 줄이는 방법
# 따라서 차원 감소를 시행할 때는 데이터의 분포를 고려하여 중요한 '축'을 찾는 일이 필요

# 특잇값 분해(Singular Value Decomposition, SVD) : 임의의 행렬을 세 행렬의 곱으로 분해하는 차원 감소 법

# 기존 PPMI 행렬을 특잇값 분해를 통해 차원 감소를 진행하는 과정
# 기존 행렬 X를 U, S, V라는 세 행렬의 곱으로 분해
# U : X 행렬의 행 크기로, 단어 공간으로 취급되는 직교 행렬
# S : 대각 행렬로, 대각 성분에 특잇값이 큰 순서로 나열
# V : X 행렬의 열 크기로, 직교 행렬

# 파이썬 코드로 SVD 구현
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

# 처리할 텍스트 문장 생성
text = 'You say goodbye and I say hello.'
# text 문장의 ID행렬, {단어 : ID} 쌍의 딕셔너리, {ID : 단어} 쌍의 딕셔너리를 전처리 함수(preprocess())를 통해 생성
corpus, word_to_id, id_to_word = preprocess(text)
# {ID : 단어} 쌍의 딕셔너리의 원소 개수를 통해 단어 개수 추출
vocab_size = len(id_to_word)
# 동시발생행렬 생성하여 C에 저장
C = create_co_matrix(corpus, vocab_size, window_size=1)
# PPMI 행렬 생성하여 W에 저장
W = ppmi(C)

# SVD를 통해 3개의 행렬로 분할 (U : W의 행 × W의 행 형상의 직교 행렬, S : 대각 성분에 특잇값을 갖는 사각 행렬, V : W의 열 × W의 열 형상의 직교 행렬)
U, S, V = np.linalg.svd(W)
# 위 SVD 연산으로 변환된 기존 행렬 W의 밀집벡터 표현은 U 행렬에 저장

# 0번째 인덱스 단어(you)의 동시발생 행렬
print(C[0])
# 0번째 인덱스 단어(you)의 PPMI 행렬
print(W[0])
# 밀집벡터의 0번째 인덱스 값
print(U[0])
# 2차원 벡터로 차원 감소한 밀집벡터(첫 2개의 원소를 꺼냄)
print(U[0,:2])

# 밀집벡터의 분포도를 2차원 그래프 상에 표시

# {word : id} 딕셔너리 쌍의 원소 쌍을 각각 word와 word_id에 입력하여 반복
for word, word_id in word_to_id.items():
    # 2차원 밀집 벡터에서 word_id에 해당하는 원소의 좌표를 2차원 평면 상에 나타내고 word(단어)로 표시
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

# 밀집벡터의 모든 원소들의 x축 지점과 y축 지점을 0.5 투명도로 점으로 표시
plt.scatter(U[:,0], U[:,1], alpha=0.5)
# 작성된 그래프 표시
plt.show()


# 관련있어 보이는 단어끼리 뭉쳐있어 우리의 직관과 유사해졌지만 사용된 말뭉치가 매우 작어서