# 일반적인 PC 실행환경 기준으로 넘파이의 부동소수점 수는 기본적으로 64비트 데이터 타입 사용

import numpy as np
# 3개의 표준 정규분포 난수(실수) 생성하여 a에 저장
a = np.random.randn(3)
# a의 데이터 타입 출력 (float64 : 64비트 부동소수점)
print(a.dtype)

# 그러나 신경망 추론 및 학습은 정확한 실수값이 필요하지 않으므로 32비트 부동소수점 수로도 문제 없이 수행 가능
# 따라서 메모리 관리 및 성능 향상의 관점에서 32비트가 더 좋으며 데이터 전송 시 버스 대역폭의 병목 방지를 위해서도 데이터 타입이 작은 것이 좋음

# 32비트 부동소수점 사용법 1 : astype 메서드의 매개변수로 'np.float32' 입력
# 3개의 표준 정규분포 난수(실수)를 32비트 부동소수점으로 생성하여 b에 저장
b = np.random.rand(3).astype(np.float32)
# b의 데이터 타입 출력 (float32 : 32비트 부동소수점)
print(b.dtype)

# 32비트 부동소수점 사용법 2 : astype 메서드의 매개변수로 'f' 입력
# 3개의 표준 정규분포 난수(실수)를 32비트 부동소수점으로 생성하여 c에 저장
c = np.random.randn(3).astype('f')
# c의 데이터 타입 출력 (float32 : 32비트 부동소수점)
print(c.dtype)

# 데이터 타입이 16비트일 경우에도 신경망 추론 및 학습에 지장없이 사용 가능
# 그러나 CPU와 GPU는 32비트 단위로 연산하므로 계산에 16비트 데이터를 사용하는 것은 속도 측면에서 혜택이 없을 수 있음
# 하지만 학습된 매개변수를 저장할 때는 메모리 용량을 절반으로 줄일 수 있으므로 데이터 저장 측면에서는 유용