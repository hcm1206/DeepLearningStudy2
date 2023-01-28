# 코사인 유사도 함수를 이용하여 검색어로 주어진 어떤 단어와 비슷한 단어를 유사도 순으로 출력하는 함수 구현

# *** 함수에서 입력받을 인수 목록 ***
# query : 검색할 단어
# word_to_id : 단어 입력받으면 ID 출력해주는 딕셔너리
# id_to_word : ID 입력받으면 단어 출력해주는 딕셔너리
# word_matrix : 단어들의 동시발생 행렬 (각 행에는 해당하는 ID의 단어가 벡터로 저장)
# top : 상위 몇 개까지의 단어를 출력할 지 설정

import numpy as np
# 코사인 유사도를 구하는 함수 불러옴
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.util import cos_similarity

# 검색할 단어(query), 단어->ID 딕셔너리(word_to_id), ID->단어 딕셔너리(id_to_word), 단어 동시발생 행렬(word_matrix), 출력할 상위 단어 수(top) 입력받음
def most_similar(query, word_to_id, id_to_word, word_matrix, top):
    # 검색할 단어가 단어 목록(word_to_id 딕셔너리의 키 값들)에 없다면
    if query not in word_to_id:
        # 검색할 단어를 찾을 수 없다고 출력
        print("%s(을)를 찾을 수 없습니다." % query)
        # 함수 종료
        return
    
    # 사용자가 입력한 검색 단어 출력
    print('\n[query] ' + query)
    # 검색 단어의 ID를 query_id에 저장
    query_id = word_to_id[query]
    # 단어 동시발생 행렬에서 검색 단어에 해당하는 벡터를 뽑아와 query_vec에 저장
    query_vec = word_matrix[query_id]

    # 문장에 존재하는 단어 개수를 vocab_size에 저장
    vocab_size = len(id_to_word)
    # 각 단어들의 검색 단어와의 유사도를 저장할 행렬을 단어 개수와 동일한 크기로 생성
    similarity = np.zeros(vocab_size)
    # 각 단어 별로 (단어 ID(i)를 통해) 반복
    for i in range(vocab_size):
        # 현재 단어 ID의 코사인 유사도를 구하여 단어 유사도 행렬의 해당 인덱스에 저장 (현재 단어의 벡터와 검색 단어의 벡터를 입력)
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    # 위 반복 과정이 끝나면 similarity 행렬에는 각 인덱스 별로 해당 ID에 해당하는 단어와 검색 단어 간의 코사인 유사도가 원소로 저장됨 
    
    # 횟수 카운트할 변수 선언
    count = 0
    # similarity(단어 유사도 행렬)을 내림차순(큰 값->작은 값)으로 정렬 후 인덱스 순서대로 반복 (argsort()는 오름차순이지만 행렬의 각 값에 -1을 곱해주면 내림차순이 됨)
    # 내림차순 정렬 후 정렬된 원소들의 인덱스 번호가 반복 변수 i의 값으로 들어감
    for i in (-1 * similarity).argsort():
        # 이번 인덱스의 단어가 검색 단어와 일치한다면
        if id_to_word[i] == query:
            # 다음 반복 실행(똑같은 단어는 포함하지 않음)
            continue
        # 현재 인덱스의 단어를 출력하고 검색 단어와의 유사도 출력
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        # 카운트 1 증가
        count += 1
        # 카운트가 최대 출력 개수(top)을 넘어섰다면
        if count >= top:
            # 함수 종료
            return



# 유사도 상위 단어 출력 함수 테스트

# 전처리 함수와 동시발생 행렬 생성 함수 불러옴
from common.util import preprocess, create_co_matrix

# 처리할 문장 text에 저장
text = 'You say goodbye and I say hello.'
# 전처리 함수를 통해 text의 단어 ID 행렬, 단어->ID 딕셔너리, ID->단어 딕셔너리 저장
corpus, word_to_id, id_to_word = preprocess(text)
# 단어->ID 딕셔너리의 원소(단어) 수를 통해 문장의 단어 개수 저장
vocab_size = len(word_to_id)
# 단어 ID 행렬, 단어 개수를 통해 text 문장의 동시발생 행렬을 생성하여 C에 저장
C = create_co_matrix(corpus, vocab_size)

# 'you' 단어와 가장 유사한 단어 상위 5개를 most_similar() 함수를 통해 출력
most_similar('you', word_to_id, id_to_word, C, top=5)
