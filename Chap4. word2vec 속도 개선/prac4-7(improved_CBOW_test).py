# 훈련된 CBOW 모델을 테스트하는 코드

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.util import most_similar
import pickle

# 매개변수가 저장된 피클 파일명을 pkl_file로 저장
pkl_file = 'files/cbow_params.pkl'

# pkl_file 파일 명의 피클 파일을 읽기용으로 불러와 저장된 딕셔너리를 params로 저장
with open(pkl_file, 'rb') as f:
    params = pickle.load(f)

    # params 딕녀서리의 'word_vecs' 키 값의 원소를 word_vecs로 저장
    word_vecs = params['word_vecs']
    # params 딕셔너리의 'word_to_id' 키 값의 원소를 word_to_id로 저장
    word_to_id = params['word_to_id']
    # params 딕셔너리의 'id_to_word' 키 값의 원소를 id_to_word로 저장
    id_to_word = params['id_to_word']

# 단어 목록으로 'you', 'year', 'car', 'toyota'를 지정하여 쿼리 querys로 저장
querys = ['you', 'year', 'car', 'toyota']
# 단어 쿼리 목록 querys에서 각 단어 쿼리 query 별로 반복
for query in querys:
    # most_similar() 메소드를 통해 해당 query 단어와 가장 유사도가 높은 상위 5개 단어 추출
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


# word2vec으로 얻은 단어 분산 표현은 단순히 비슷한 단어를 찾을 뿐 아니라 복잡한 패턴도 파악 가능

# 유추(비유) 문제 : 어떤 단어에서 특정 속성 단어를 뺀 의미의 다른 단어를 추측 가능
# 즉 단어 벡터의 덧셈과 뺄셈으로 단어 유추 문제를 해결 가능


# 이러한 유추 로직을 구현한 analogy() 메소드 사용

from common.util import analogy
# 'man':'king'의 관계와 유사한 'woman'과 관계를 갖는 단어 유추 (유사한 의미의 단어 관계)
analogy('man', 'king', 'woman', word_to_id, id_to_word, word_vecs)
# 'take':'took'의 관계와 유사한 'go'와 관계를 갖는 단어 유추 (과거형 시제의 단어 관계)
analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
# 'car':'cars'의 관계와 유사한 'child'와 관계를 갖는 단어 유추 (복수형 단어 관계)
analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
# 'good':'better'의 관계와 유사한 'bad'와 관계를 갖는 단어 유추 (비교급 단어 관계)
analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)

# 위 예시에서는 단어 관계를 잘 유추한 것으로 보이지만
# 말뭉치 크기가 작아서 다른 유추 관계를 찾아내는데 성능이 좋지 않음
# 말뭉치 크기를 충분히 키우면 유추 문제의 정확도 향상 기대 가능