import numpy as np

# x1, x2 두개의 입력노드가 AND 연산의 결과를 출력해야 한다고 했을 때
# 퍼셉트론을 작성해보자.

def create_perceptron_simple(w1, w2, theta):
    def perceptron(x1, x2):
        return 1 if x1 * w1 + x2 * w2 > theta else 0
    
    return perceptron

AND_simple = create_perceptron_simple(0.5, 0.5, 0.7)

print("- 퍼셉트론")
print(AND_simple(0, 0)) # 0
print(AND_simple(0, 1)) # 0
print(AND_simple(1, 0)) # 0
print(AND_simple(1, 1)) # 1

# x1 * w1 + x2 * w2 > theta
# => b + x1 * w1 + x2 * w2 > 0 (theta 를 b로 변경)
# => b는 편향(bias)를 의미함
def create_perceptron(wnp, bias):
    def perceptron(xnp):
        return 1 if np.sum(wnp * xnp) + bias > 0 else 0
    
    return perceptron

# AND  : 0,0 => 0 / 0,1 => 0 / 1,0 => 0 / 1,1 => 1
# NAND : 0,0 => 1 / 0,1 => 1 / 1,0 => 1 / 1,1 => 0
# OR   : 0,0 => 0 / 0,1 => 1 / 1,0 => 1 / 1,1 => 1
AND = create_perceptron(np.array([0.5, 0.5]), -0.7)
NAND = create_perceptron(np.array([-0.5, -0.5]), 0.7)
OR = create_perceptron(np.array([0.5, 0.5]), -0.2)

print("- AND")
print(AND(np.array([0, 0])))
print(AND(np.array([0, 1])))
print(AND(np.array([1, 0])))
print(AND(np.array([1, 1])))
print("- NAND")
print(NAND(np.array([0, 0])))
print(NAND(np.array([0, 1])))
print(NAND(np.array([1, 0])))
print(NAND(np.array([1, 1])))
print("- OR")
print(OR(np.array([0, 0])))
print(OR(np.array([0, 1])))
print(OR(np.array([1, 0])))
print(OR(np.array([1, 1])))

# 다중 퍼셉트론으로 xor 게이트 구현하기
# XOR  : 0,0 => 0 / 0,1 => 1 / 1,0 => 1 / 1,1 => 0
# XOR = (NAND,OR => AND)
def XOR(xnp):
    return AND([np.array([NAND(xnp), OR(xnp)])])

print("- XOR")
print(XOR(np.array([0, 0])))
print(XOR(np.array([0, 1])))
print(XOR(np.array([1, 0])))
print(XOR(np.array([1, 1])))