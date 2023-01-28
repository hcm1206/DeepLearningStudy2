# 단어를 RGB 색상 표기법처럼 벡터(다차원 행렬)로 표현하여 단어의 의미를 파악할 수 있도록 하는 방법을 분산 표현(distributional representaion)이라 함

# 통계 기반 기법은 단어의 의미가 주변 단어들의 분포에 의해 형성된다는 기본 아이디어를 바탕으로 구현되고 있음
# 즉 단어 자체에서 의미를 찾기보다 단어가 사용된 맥락에서 의미를 찾는 것
# 윈도우 크기 : 맥락의 크기, 즉 기준이 되는 특정 단어의 앞뒤로 몇 개의 단어를 포함할 것인가를 결정하는 요소
# 이런 방식으로 특정 단어 주변에 어떤 단어가 몇 번 등장하는지 세어 집계하는 분포 가설 방법을 '통계 기반(statistical based) 기법'이라 함

# =============== 통계 기반 기법 예시 ==================

# --------------- 데이터 전처리 -------------------
# 전처리 과정은 prac2-2 내용과 동일
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

# text 문장의 단어를 ID로 바꾼 벡터 출력
print(corpus)
# {ID : 단어} 쌍의 딕셔너리 출력
print(id_to_word)

print()

# ------------------ 단어 맥락으로 동시에 발생(등장)하는 단어 빈도 체크 ----------------------
# 문장에서 단어 'you'의 맥락은 'say'라는 단어 하나 뿐
# 'say'는 ID가 1이므로 'you'의 맥락에 포함되는 단어의 빈도를 벡터로 표현하면
# [0,1,0,0,0,0,0]와 같이 표현됨

# 단어 'say'의 맥락은 'you'와 'goodbye' 두 가지
# 'you'는 ID가 0이고 'goodbye'는 ID가 2
# 단어 빈도 벡터는 [1,0,1,0,0,0,0]으로 표현

# 이러한 단어 별 단어빈도벡터를 표로 나타낸 것을 동시발생 행렬(co-occurance matrix)이라 함

# text의 동시발생 행렬을 위와 같은 과정으로 수동으로 구현
C = np.array([
    [0,1,0,0,0,0,0], # you(ID : 0)의 단어 빈도 벡터
    [1,0,1,0,0,0,0], # say(ID : 1)의 단어 빈도 벡터
    [0,1,0,1,0,0,0], # goodbye(ID : 2)의 단어 빈도 벡터
    [0,0,1,0,1,0,0], # I(ID : 3)의 단어 빈도 벡터
    [0,1,0,1,0,0,0], # say(ID : 4)의 단어 빈도 벡터
    [0,1,0,0,0,0,1], # hello(ID : 5)의 단어 빈도 벡터
    [0,0,0,0,0,1,0], # .(ID : 6)의 단어 빈도 벡터
], dtype=np.int32) # 32비트 정수로 데이터 타입 사용

# 각 단어의 벡터 획득 방법
print("C[0] :", C[0]) # ID가 0인 단어(you)의 벡터 출력
print("C[4] :", C[4]) # ID가 4인 단어(say)의 벡터 출력
print("goodbye :", C[word_to_id['goodbye']]) # 'goodbye' 단어의 벡터 출력

# 말뭉치를 입력하여 동시발생 행렬을 자동으로 만들어주는 함수 구현
# corpus(문장의 ID 배열, 단어 개수, 윈도우 크기(맥락에 사용할 주변 단어 범위, 기본값 1)를 입력받음)
def create_co_matrix(corpus, vocab_size, window_size=1):
    # 문장 ID 배열의 크기를 corpus_size에 저장
    corpus_size = len(corpus)
    # 동시발행 행렬을 (단어 개수) × (단어 개수) 형상의 행렬로 생성 (데이터 타입은 32비트 정수로 사용)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    # corpus의 각 원소 별로 반복하여 idx로 현재 반복 중인 ID를, word_id로 원소(현재 단어의 ID)를 사용
    for idx, word_id in enumerate(corpus):
        # 1부터 윈도우 크기 미만까지 반복 변수 i로 반복 (윈도우 크기가 n이면 그만큼 좌측 인덱스와 우측 인덱스의 개수도 n개)
        for i in range(1, window_size + 1):
            # 좌측 인덱스를 현재 인덱스(idx)에서 반복 횟수(i)만큼 뺀 값으로 설정
            left_idx = idx - i
            # 우측 인덱스를 현재 인덱스(idx)에서 반복 횟수(i)만큼 더한 값으로 설정
            right_idx = idx + i

            # 이번 좌측 인덱스가 0 이상이면 (인덱스 범위를 벗어나지 않았다면)
            if left_idx >= 0:
                # 좌측 단어 id를 corpus의 이번 좌측 인덱스 값(좌측 인덱스에 해당하는 단어 ID)로 설정
                left_word_id = corpus[left_idx]
                # 동시발생 행렬의 [(현재 주목 중인 단어 ID), (이번 좌측 단어 ID)] 원소에 1을 더하여 저장
                co_matrix[word_id, left_word_id] += 1
            
            # 이번 우측 인덱스가 문장의 단어 개수 미만이면 (인덱스 범위를 벗어나지 않았다면)
            if right_idx < corpus_size:
                # 우측 단어 id를 corpus의 이번 우측 인덱스 값(우측 인덱스에 해당하는 단어 ID)로 설정
                right_word_id = corpus[right_idx]
                # 동시발생 행렬의 [(현재 주목 중인 단어 ID), (이번 우측 단어 ID)] 원소에 1을 더하여 저장
                co_matrix[word_id, right_word_id] += 1
            # 여기까지 진행하면 현재 word_id에 해당하는 단어의 벡터가 생성됨(맥락에 해당하는 단어 ID 부분은 1, 아닌 부분은 0으로 표현)

        # 모든 word_id에 해당하는 단어 벡터가 완성되면 동시발생 행렬 생성이 완료됨

    # 생성한 동시발생 행렬 리턴
    return co_matrix

# 동시발생 행렬 자동 생성 함수 테스트
# 기존과 동일한 전처리 된 데이터의 단어 ID 배열, 문장의 단어 개수(word_to_id 행렬의 길이 사용)를 동시발생 행렬 함수에 입력한 결과를 func_co_matrix에 저장
func_co_matrix = create_co_matrix(corpus, len(word_to_id))
# 생성된 동시발생 행렬이 저장된 func_co_matrix의 행렬 출력
print("함수로 생성한 동시발생 행렬 : \n", func_co_matrix)

# 결과는 수동으로 만들었던 동시발생 행렬과 동일
# 동시발생 행렬 자동 생성 함수는 common/util.py에도 별도로 저장되어 있음