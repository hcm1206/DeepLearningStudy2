# 벡터로 나타낸 단어 사이의 유사도를 측정하는 방법은 다양하지만 대표적으로 코사인 유사도(cosine similarity) 사용

# 두 벡터 x = (x1, x2, x3, ..., xn)과 y = (y1, y2, y3, ..., yn)가 있을 때
# 코사인 유사도는 분자에 벡터의 내적을, 분모에 각 벡터의 노름이 들어간 분수 형태로 표현됨

# 노름 : 벡터의 크기를 나타낸 것
# 우리가 사용할 노름은 L2 노름으로, 벡터의 각 원소를 제곱하여 더한 후 다시 제곱근을 구하여 계산하는 방식
# 핵심은 벡터를 정규화하고 내적을 구하는 것

# 코사인 유사도는 두 벡터가 가리키는 방향이 얼마나 비슷한가를 나타내며, 방향이 완전히 같다면 1, 정반대 방향이면 -1이 됨



# ============================== 코사인 유사도를 구하는 함수 구현 ==============================

import numpy as np

# x(단어) 벡터와 y(단어) 벡터를 입력받음
def cos_similarity(x, y):
    # x 값을 분자로 하고 x 벡터의 원소들을 제곱하여 모두 더한 값에 제곱근을 취한 값을 분모로 하여 nx값 구함 (x의 정규화)
    nx = x / np.sqrt(np.sum(x**2))
    # y 값을 분자로 하고 y 벡터의 원소들을 제곱하여 모두 더한 값에 제곱근을 취한 값을 분모로 하여 ny값 구함(y의 정규화)
    ny = y / np.sqrt(np.sum(y**2))
    # nx값과 ny값을 행렬곱한 최종 계산 결과 리턴
    return np.dot(nx, ny)

# 위 함수는 인자로 들어오는 단어 벡터가 모두 0인 경우 0으로 나누기 오류가 발생하는 문제가 있음
# 따라서 부동소수점 계산 시 영향을 끼치지 않을 정도의 작은 수를 분모에 더해주어 0으로 나눠지는 경우가 없도록 수정


#  ============================== 0으로 나누기 오류 수정한 함수 구현 ==============================
# 위와 같은 코사인 유사도를 구하는 동일한 내용의 함수이지만 아주 작은 값인 eps 값을 분모에 추가하여 0으로 나누기 오류 해결한 함수 구현

# x(단어) 벡터와 y(단어) 벡터를 입력받고 분모에 추가할 아주 작은 값 eps(기본값 1e-8 = 0.00000001)도 입력받음
def cos_similarity(x, y, eps=1e-8):
    # 분모에 eps 값 더함
    nx = x / (np.sqrt(np.sum(x**2))) + eps
    # 분모에 eps 값 더함
    ny = y / (np.sqrt(np.sum(y**2))) + eps
    return np.dot(nx, ny)

# 위 함수는 common/util.py 파일에도 별도로 구현되어있음

# ============================== 코사인 유사도 함수 테스트 ==============================

# 단어 벡터의 유사도 구하는 예시 : you와 i의 유사도를 구함

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# 전처리 함수와 동시발생 행렬 생성 함수 불러옴
from common.util import preprocess, create_co_matrix

# 처리할 문장을 text에 저장
text = 'You say goodbye and I say hello.'
# preprocess 함수를 통해 text 문장을 통해 단어 ID 배열, {단어 : ID} 쌍의 딕셔너리, {ID : 단어} 쌍의 딕셔너리 생성
corpus, word_to_id, id_to_word = preprocess(text)
# 단어 개수를 word_to_id 딕셔너리의 원소 개수를 통해 구함
vocab_size = len(word_to_id)
# 단어 ID 배열과 단어 개수를 통해 text 문장의 동시발생 행렬 생성하여 C에 저장
C = create_co_matrix(corpus, vocab_size)

# text 문장의 동시발생 행렬에서 'you' 단어의 맥락 빈도 벡터를 구하여 c0에 저장
c0 = C[word_to_id['you']]
# text 문장의 동시발생 행렬에서 'i' 단어의 맥락 빈도 벡터를 구하여 c1에 저장
c1 = C[word_to_id['i']]
# c0(you) 행렬과 c1(i) 행렬의 코사인 유사도를 구하여 출력
print("you와 i의 코사인 유사도 :", cos_similarity(c0, c1))

# 코사인 유사도 결과는 0.70으로 나오며, 코사인 유사도 값은 최소 -1, 최대 1이므로 이 값은 유사성이 비교적 크다고 볼 수 있음
