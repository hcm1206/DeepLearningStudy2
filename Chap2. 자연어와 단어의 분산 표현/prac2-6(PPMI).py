# prac2-5에서 사용한 동시발생 행렬에서 발생 횟수는 고빈도 단어로 인하여 사용하기 좋은 특징이 아님
# 예를 들어 the와 같이 불특정 다수의 단어와 함께 자주 사용되는 단어의 경우 동시발생률이 실제 의미가 유사한 단어보다 더 자주 나타남

# 이를 해결하기 위하여 점별 상호정보량(Pointwise Mutual Information, PMI) 척도 사용
# 어떤 두 단어 x, y에 대하여 PMI는 x와 y가 동시에 일어날 확률을 x가 일어날 확률과 y가 일어날 확률의 곱으로 나누고 로그 2를 취한 값이 x와 y의 PMI

# 이를 동시발생 행렬을 사용하여 정의하면 아래와 같음
# x와 y가 발생하는 횟수를 각각 C(x), C(y), x와 y가 동시에 발생하는 횟수를 C(x,y), 말뭉치에 포함된 모든 단어의 수를 N이라 하면
# PMI는 C(x,y)와 N의 곱을 C(x)와 C(y)의 곱으로 나누고 로그 2를 취한 값

# PMI는 두 단어의 동시발생 횟수(C(x,y))가 0이면 로그 2의 0이 되어 무한대 음수가 된다는 문제가 있음
# 이 문제를 피하기 위해 양의 상호정보량(Positive PMI, PPMI) 사용
# PPMI는 PMI가 음수일 때 결과를 0으로 바꿔주기만 하면 됨


# PPMI 행렬 구하는 함수 구현
import numpy as np

# 동시발생 행렬 C, 진행상황 출력 여부(verbose, 기본값 False), 오류 제어를 위한 아주 작은수 (eps, 기본값 1e-8 = 0.00000001) 입력
def ppmi(C, verbose=False, eps=1e-8):
    # 동시발생 행렬 C와 같은 형상이고 데이터 타입이 32비트 실수인 행렬을 생성하여 M에 저장 (PPMI 행렬을 저장할 변수)
    M = np.zeros_like(C, dtype=np.float32)
    # 동시발생 행렬 C의 모든 원소의 합을 구하여 모든 단어의 수를 N에 저장
    N = np.sum(C)
    # 동시발생 행렬 C에서 세로(0차원)축으로 모든 원소를 더하여 각 단어 별 발생 빈도 값을 S에 저장
    S = np.sum(C, axis=0)
    # 동시발생 행렬 C에서 세로(0차원)축(각 단어)과 가로(1차원)축(현재 단어)를 곱하여 각 단어에 대응하는 단어들의 전체 수를 구함
    total = C.shape[0] * C.shape[1]
    # 횟수 변수 선언하여 0 저장
    cnt = 0

    # 각 단어(i) 별로 반복
    for i in range(C.shape[0]):
        # 현재 단어(i)에서 대응하는 각 단어(j) 별로 반복
        for j in range(C.shape[1]):
            # 현재 단어(i)와 대응하는 단어(j)와의 PMI 계산
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            # PMI가 음수이면 0으로 설정하고 최종 값을 PPMI값으로 대응되는 행렬 M의 알맞은 위치에 저장
            M[i, j] = max(0, pmi)

            # 진행상황 출력 여부가 True라면
            if verbose:
                # 카운트를 1 올림
                cnt += 1
                # 전체 진행 상황의 백분율 값이 정수로 1% 이상 증가하였다면
                if cnt % (total//100 + 1) == 0:
                    # 현재 진행 상황 백분율 출력
                    print('%.1f%% 완료' % (100*cnt/total))

    # 최종 PPMI 행렬 M 리턴
    return M


# PPMI 행렬 구하는 함수를 이용하여 동시발생 행렬을 PPMI 행렬로 변환하기

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
# 전처리 함수, 동시발생 행렬 생성 함수 불러옴
from common.util import preprocess, create_co_matrix, cos_similarity

# 처리할 문장 하나를 text에 저장
text = 'You say goodbye and I say hello.'
# text 문장의 ID행렬, {단어:ID} 쌍의 딕셔너리, {ID:단어} 쌍의 딕셔너리를 전처리 함수를 통해 구함
corpus, word_to_id, id_to_word = preprocess(text)
# {단어:ID} 쌍의 딕셔너리의 원소 수를 통해 text의 단어 개수를 구하여 vocab_size로 저장
vocab_size = len(word_to_id)
# text 문장의 단어 ID 행렬과 단어 개수를 통해 동시발생 행렬을 구하여 C에 저장
C = create_co_matrix(corpus, vocab_size)
# 동시발생 행렬 C의 PPMI 행렬을 구하여 W에 저장
W = ppmi(C)

# 넘파이 배열에서 소수점을 3자리까지 반올림하는 것으로 설정
np.set_printoptions(precision=3)
# 동시발생 행렬 출력
print('동시발생 행렬')
print(C)
print('-'*50)
# PPMI 행렬 출력
print('PPMI')
print(W)


# PPMI 행렬의 문제점
# 1. 말뭉치 어휘 수에 따라 각 단어 벡터의 차원 수도 그만큼 증가 (벡터 차원이 기하급수적으로 커짐)
# 2. 행렬의 원소 대부분이 0으로, 중요하지 않은 정보가 벡터의 대부분을 차지