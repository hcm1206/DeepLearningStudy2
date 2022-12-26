# 벡터와 행렬

# 벡터와 행렬 데이터를 다루기 위해 numpy 라이브러리 사용
import numpy as np

# x에 [1,2,3] numpy 배열 저장
x = np.array([1,2,3])
# x 변수의 클래스 타입 출력 (numpy.ndarray : numpy의 ndarray 클래스)
print(x.__class__)
# x 변수의 배열 형상 출력 (3, : 3×1 형상의 배열)
print(x.shape)
# x 변수의 배열 차원 출력 (1 : 1차원)
print(x.ndim)

# x에는 1차원이고 원소 수가 3개인 배열 저장
# x는 벡터이자 1차원 행렬로 볼 수 있음

# W에 [[1,2,3],[4,5,6]] numpy 배열 저장
W = np.array([[1,2,3],[4,5,6]])
# W 변수의 배열 형상 출력 (2,3 : 2×3 형상의 배열)
print(W.shape)
# W 변수의 배열 차원 출력 (2, 2차원)
print(W.ndim)

# W에는 2차원이고 원소 수가 각 행당 3개인 배열 저장
# W는 2차원 행렬로 볼 수 있음