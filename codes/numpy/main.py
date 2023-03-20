import numpy as np

print("numpy 기본")

array_a = np.array([1.0, 2.0, 3.0])
print("# 배열")
print(array_a)

array_b = np.array([2.0, 4.0, 6.0])
print("# 배열 연산")
print(array_a + array_b)
print(array_a - array_b)
print(array_a / array_b)
print(array_a / 2)

matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[3, 0], [0, 6]])
print("# 행렬")
print(matrix_a)
print(matrix_b)
print(matrix_a.shape)
print(matrix_a.dtype)

print("# 행렬 연산")
print(matrix_a + matrix_b)
print(matrix_a * matrix_b)

matrix_c = np.array([10, 20])
print("# 브로드캐스트")
print("2x2 행렬과 1x2 행렬을 곱연산이 동작했다.")
print("이는 numpy 에서 알아서 스칼라 곱으로 인식해서 [[10, 20],[10,20]] 형태로 스칼라 곱을 해주기 때문이다.")
print(matrix_a * matrix_c)

print("# 평탄화")
flattened = matrix_a.flatten()
print(flattened)

print("# 인덱스 필터")
print("0, 2 인덱스만 필터링")
print(flattened[np.array([0, 2])])

print("# 조건으로 변환")
print(flattened > 2)

print("# 조건으로 필터")
print("짝수만 필터")
print(flattened[flattened % 2 == 0])
