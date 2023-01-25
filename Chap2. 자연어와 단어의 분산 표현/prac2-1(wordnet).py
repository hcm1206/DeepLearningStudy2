# 자연어 처리 방법 중 하나는 미리 단어들을 유의어들끼리 관계를 갖는 시소러스 형태의 단어 사전을 정의하여 사용하는 것
# 시소러스는 단어 사이의 상하위 포함 개념, 전체와 부분 개념, 유의어 관계 개념 등 각 단어들과의 관계를 정의해 놓는 경우가 많음
# 이러한 대규모 단어 네트워크가 저장된 일종의 데이터베이스 역할을 하는 컴퓨터용 단어 사전을 시소러스라고 함

# 자연어 처리 분야에서 가장 유명한 시소러스는 WordNet
# WordNet을 파이썬으로 다뤄보는 실습 진행

# WordNet을 사용하기 위한 nltk 라이브러리 설치
# 터미널에서 pip install nltk 작성

# nltk 라이브러리 불러옴
import nltk
# 최초 실행 시에는 아래 코드를 실행하여 wordnet 데이터를 다운로드해야 함
# nltk.download('wordnet')

# nltk 라이브러리에서 wordnet 객체 불러옴
from nltk.corpus import wordnet

# wordnet의 car 동의어 데이터 출력 : 원소 5개짜리 리스트가 출력됨(car 단어에 5가지 의미(동의어 그룹)이 존재한다는 뜻)
print(wordnet.synsets('car'))

# car에 car.n.01 동의어 그룹 저장
car = wordnet.synset('car.n.01')
# car(car.n.01)에 저장된 동의어 그룹의 정의 출력 : car의 1번째 정의 출력
print(car.definition())

# car(car.n.01) 표제어의 동의어 그룹에 존재하는 단어 목록 출력 : '자동차'라는 뜻의 car과 유사한 뜻을 지닌 동의어 목록 리스트 출력
print(car.lemma_names())


# WordNet에서 각 단어들은 의미적인 상하관계 네트워크가 구축되어 있어 어떤 단어는 다른 단어의 부분집합이라는 것을 확인할 수 있음

# car(car.n.01) 표제어에서 네트워크로 연결된 다른 단어와의 의미적인 상하관계 출력 : entity->physical_entitiy->object-> ... ->motor_vehicle->car 순으로 상하관계 형성
print(car.hypernym_paths()[0])


# WordNet에서 단어 간 의미 네트워크를 통해 각 단어가 의미적으로 얼마나 유사한지 측정하는 방법 존재, 1에 가까울수록 의미가 유사한 단어라는 뜻

# novel에 novel.n.01(소설) 동의어 그룹 저장
novel = wordnet.synset('novel.n.01')
# dog에 dog.n.01(개) 동의어 그룹 저장
dog = wordnet.synset('dog.n.01')
# motorcycle에 motorcycle.n.01(오토바이) 동의어 그룹 저장
motorcycle = wordnet.synset('motorcycle.n.01')

# car(자동차)와 novel(소설) 간의 의미 유사도 출력 : 매우 적은 유사도 (0.0555...)
print(car.path_similarity(novel))
# car(자동차)와 dog(개) 간의 의미 유사도 출력 : 적은 유사도 (0.0769...)
print(car.path_similarity(dog))
# car(자동차)와 motorcycle(오토바이) 간의 의미 유사도 출력 : 다소 큰 유사도 (0.3333...)
print(car.path_similarity(motorcycle))

# 이러한 유사도 네트워크를 통해 각 단어의 의미를 컴퓨터가 간접적으로 해석 가능

