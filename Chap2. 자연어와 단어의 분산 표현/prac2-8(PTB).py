# 팬 트리뱅크(PTB, Penn TreeBank) : 적당한 크기의 자연어 처리 기법 벤치마킹용 말뭉치
# PTB 데이터셋은 dataset/ptb.py 파일을 통해 적절하게 불러올 수 있음

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# ptb 데이터셋 관련 함수가 저장된 파일 불러오기
from dataset import ptb

# ptb 데이터셋을 훈련용으로 불러와 ptb 데이터셋 문장의 ID 행렬, ptb의 {단어 : ID} 쌍의 딕셔너리, ptb의 {ID, 단어} 쌍의 딕셔너리 저장
corpus, word_to_id, id_to_word = ptb.load_data('train')

# PTB 데이터셋 사용법 예시

# ID행렬의 크기를 통해 말뭉치 크기 측정하여 출력
print('말뭉치 크기:', len(corpus))
# 0번째부터 29번째 ID 행렬에 저장된 ID 출력
print('corpus[:30]:', corpus[:30])
print()
# 0번 ID의 단어 출력
print('id_to_word[0]:', id_to_word[0])
# 1번 ID의 단어 출력
print('id_to_word[1]:', id_to_word[1])
# 2번 ID의 단어 출력
print('id_to_word[2]:', id_to_word[2])
print()
# car 단어의 ID 출력
print("word_to_id['car']:", word_to_id['car'])
# happy 단어의 ID 출력
print("word_to_id['happy']:", word_to_id['happy'])
# lexus 단어의 ID 출력
print("word_to_id['lexus']:", word_to_id['lexus'])