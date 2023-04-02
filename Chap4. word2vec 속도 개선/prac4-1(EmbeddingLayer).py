# word2vec 속도 개선 방법 1
# 어휘 수가 거대해지면 입력층의 원핫 표현 벡터 길이가 크게 증가하여 가중치 행렬 Win의 행렬곱 계산이 매우 복잡해짐
# 이 문제를 해결하기 위해 Embedding 계층 도입

# 입력층에서 은닉층으로의 가중치 계산은 결과적으로 가중치 행렬의 특정 행을 추출하는 것
# 따라서 거대한 크기의 행렬곱 계산을 일일이 수행할 필요가 없음 
# 원핫 표현으로의 변환과 MatMul계층의 행렬곱 계산을 할 필요 없이 Embedding 계층으로 통합 가능
# Embedding(임베딩) : 단어 임베딩에서 유래한 표현으로, 단어의 밀집벡터 표현을 일컫는 용어

# 임베딩을 통한 단어 밀집벡터 추출 예시
import numpy as np
# 어휘 수가 7, 은닉층이 3인 CBOW 모델에서의 7×3 형상의 가중치(단어밀집) 행렬를 생성하여 W에 저장
W = np.arange(21).reshape(7,3)
# 가중치 행렬 W의 전체 원소 출력
print("단어 밀집벡터 표현이 저장된 가중치 W 행렬 : ")
print(W)
# W 행렬의 2번 인덱스에 저장된 단어의 밀집벡터 출력
print("인덱스 2번 단어의 밀집벡터 :", W[2])
# W 행렬의 5번 인덱스에 저장된 단어의 밀집벡터 출력
print("인덱스 5번 단어의 밀집벡터 :", W[5])

# 여러 개의 인덱스가 담긴 넘파이 배열 idx 생성
idx = np.array([1, 0, 3, 0])
# W 행렬에서 idx의 원소에 해당하는 단어밀집벡터를 각각 생성 (여러 행을 동시에 추출 가능)
print(W[idx])

# Embedding 계층을 클래스로 구현
class Embedding:
    # 클래스 생성자(인자로 가중치 행렬 W 입력받음)
    def __init__(self, W):
        # 계층 매개변수로 W 행렬을 행렬 원소로 하여 params로 저장
        self.params = [W]
        # 계층 기울기로 W 행렬과 같은 형상의 0으로 이루어진 행렬을 행렬 원소로 하여 grads로 저장
        self.grads = [np.zeros_like(W)]
        # 처리할 인덱스를 맴버 변수 idx로 선언
        self.idx = None
    
    # 순전파 계산(인자로 처리할 행렬 인덱스 idx를 입력받음)
    def forward(self, idx):
        # params(가중치 매개변수) 맴버 변수의 값을 W에 저장
        W, = self.params
        # 입력받은 처리할 행렬 인덱스 idx를 맴버 변수 idx에 저장
        self.idx = idx
        # W 행렬의 idx에 해당하는 원소를 출력값 out에 저장
        out = W[idx]
        # 출력값 out 리턴
        return out
    
        # 순전파 계산에서는 입력층(W)에서의 특정(idx) 행을 그대로 출력층으로 보냄
    
    # 역전파 계산(인자로 역전파 입력값 dout을 입력받음)
    def backward(self, dout):
        # grads(매개변수 기울기) 맴버 변수의 값을 dW에 저장
        dW, = self.grads
        # dW의 모든 원소에 0 저장
        dW[...] = 0

        # idx 행렬의 인덱스(i), 원소(word_id)를 반복변수로 하여 반복
        for i, word_id in enumerate(self.idx):
            # dW 행렬의 word_id 인덱스에 해당하는 원소(입력된 인덱스 위치)에 dout 원소의 i번째 인덱스(역전파 행렬에서 해당되는 위치)의 값을 더하여 저장
            dW[word_id] += dout[i]

            # 위 코드는 아래와 같이 변경 가능(dW 행렬의 맴버변수 idx 위치에 동일한 idx 위치의 dout 원소를 추가한다는 뜻)
            #np.add.at(dW, self.idx, dout)
        
        # 아무것도 리턴하지 않음
        return None
    
        # 순전파 계산에서는 W 행의 특정 행을 단순히 추출하는 것이었으므로 역전파 계산에서는 역전파 입력값(출력층)에서 받은 기울기(dout)를 그대로 특정(idx) 행으로 보내주기만 하면 됨
        # 이 때 dW 행렬에 해당되는 원소를 할당(=)하는 것이 아닌 더하는(+=) 이유는 중복되는 원소가 idx 행렬에 존재할 때 할당을 하면 덮어씌워지는 문제가 발생하기 때문