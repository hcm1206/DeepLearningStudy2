# 통계 기반 기법에 사용되는 말뭉치 : 일반적으로 대량의 텍스트 데이터이고, 자연어 처리 연구나 애플리케이션을 염두해 두고 수집된 텍스트 데이터
# 이러한 말뭉치에서 컴퓨터가 자동적이고 효율적으로 텍스트의 핵심과 의미를 추출하는 것이 통계 기반 기법의 목표

# ================== 단순한 하나의 문장으로 이루어진 텍스트를 가공하기 위한 전처리 과정 ========================

# 하나의 문장으로 이루어진 텍스트 데이터를 text에 저장
text = 'You say goodbye and I say hello.'
# 텍스트 데이터 문장 출력
print("원본 text :", text)
# ----------------- 1. 텍스트를 단어 단위로 분할 -------------------------

# 텍스트 데이터의 모든 문자를 소문자로 변환
text = text.lower()
# 텍스트 데이터에서 온점(.)을 단어에서 모두 띄움(단어와 온점 사이에 띄어쓰기 공백 한 칸 생성)
text = text.replace('.', ' .')
# 단어 별로 분할된 텍스트 데이터 출력
print("분할된 text :", text)

# 분할된 텍스트 데이터에서 띄어쓰기 공백을 기준으로 분할된 단어들을 각각 원소로 저장한 리스트를 words에 저장
words = text.split(' ')
# 분할된 단어들이 원소로 저장된 리스트 words 출력
print("text의 단어 리스트 words :", words)

print()

# ------------------- 2. 분할된 단어에 ID를 부여하여 딕셔너리 형태로 가공 ----------------------

# 단어(word)를 key로 입력하면 해당 단어의 ID를 출력하게 할 딕셔너리를 생성
word_to_id = {}
# ID를 key로 입력하면 해당 ID의 단어(word)를 출력하게 할 딕셔너리를 생성
id_to_word = {}

# 단어 목록 리스트에서 각 단어 별로 반복
for word in words:
    # word_to_id 딕셔너리에 현재 word가 원소로 존재하지 않을 떄
    if word not in word_to_id:
        # word_to_id 딕셔너리의 현재 원소 수를 새로운 ID 값으로 생성(원소(ID) 수가 n개면 마지막 ID는 n-1이므로 새 ID는 n이 됨)
        new_id = len(word_to_id)
        # word_to_id 딕셔너리에서 word 키 값에 대응하는 원소(ID)를 새 ID(new_id) 값으로 하여 저장
        word_to_id[word] = new_id
        # id_to_word 딕셔너리에서 새 ID(new_id) 키 값에 대응하는 원소(word)를 새 word(단어)로 하여 저장
        id_to_word[new_id] = word

# id_to_word ({ID : 단어} 쌍의 딕셔너리 출력)
print("id_to_word :", id_to_word)
# word_to_id ({단어 : ID} 쌍의 딕셔너리 출력)
print("word_to_id :", word_to_id)

# 딕셔너리에서 ID를 이용하여 단어 검색 또는 단어를 통해 ID 검색 가능 
# id_to_word 딕셔너리에서 key가 1인 원소(ID가 1인 단어) 출력
print("ID가 1인 단어 :", id_to_word[1])
# word_to_id 딕셔너리에서 key가 'hello'인 원소(hello의 ID) 출력
print("'hello' 단어의 ID :", word_to_id['hello'])

print()

# ------------------- 3. 단어 목록을 단어 ID 목록으로 변경 ----------------------
# 문장에서 단어를 각 원소로 분할한 리스트인 words에서 원래 단어가 저장된 원소를 해당 단어의 ID 값으로 변경하는 작업

import numpy as np
# words 리스트에서 각 원소(단어) 별로 반복하여 해당 단어의 ID를 추출한 후 해당 ID를 corpus 리스트에 차례대로 원소로 저장 (파이썬 내포(comprehension) 표기 이용)
corpus = [word_to_id[w] for w in words]
# corpus 리스트를 넘파이 배열로 변환
corpus = np.array(corpus)
# corpus 배열 출력
print("corpus :", corpus)

# 단어가 원소로 저장되어있던 리스트가 해당 단어의 ID가 원소로 저장된 리스트로 바뀜

print()

# ======================= 위의 모든 전처리 과정을 하나의 함수로 구현 =========================
# 텍스트 데이터(text) 입력받는 전처리 함수 정의
def preprocess(text):
    # 문장에서 단어 분할
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    
    # 단어 별 ID 할당 후 딕셔너리 생성
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # 단어를 해당 단어의 ID로 변환한 배열 생성
    corpus = np.array([word_to_id[w] for w in words])

    # 텍스트 데이터의 ID 배열(corpus), {단어 : ID} 쌍의 딕셔너리(word_to_id), {ID : 단어} 쌍의 딕셔너리(id_to_word) 리턴
    return corpus, word_to_id, id_to_word

# 전처리 함수 테스트
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print("전처리 함수 corpus :", corpus)
print("전처리 함수 word_to_od :", word_to_id)
print("전처리 함수 id_to_word :", id_to_word)

# 전처리 함수는 common/util.py 파일에도 저장되어 있음