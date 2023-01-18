# 지금까지 구현한 신경망을 사용하기 위하여 데이터셋 준비

import sys, os
# 상위 디렉토리의 파일에 접근할 수 있도록 설정 (Chap1 디렉토리의 상위 디렉토리인 DeepLearningStudy2 디렉토리의 파일 접근 가능)
# sys.path.append('..') 가 원본코드이지만 아나콘다 환경에서는 작동하지 않으므로 아래 코드 사용
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# dataset 디렉토리의 spiral 파일에서 데이터셋을 불러올 예정
from dataset import spiral

import matplotlib.pyplot as plt

# spiral 데이터셋에서 입력 데이터 x와 정답 레이블 데이터 t를 받아옴
x, t = spiral.load_data()
# 입력 데이터 x의 형상 출력
print('x', x.shape)
# 정답 레이블 데이터 t의 형상 출력
print('t', t.shape)

# 데이터셋의 산점도 그리기
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N,0], x[i*N:(i+1)*N,1], s=40, marker=markers[i])
plt.show()

# 산점도를 확인하면 데이터셋의 데이터가 3가지로 구분되며 나선형 형태로 분포하고 있음