import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return 1 if x > 0 else 0

def step_function_using_numpy(x):
    y = x > 0
    return y.astype(int)

print(step_function(-1))
print(step_function_using_numpy(np.array([1, 2, 0, -1])))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(np.array([-1.0, 1.0, 2.0])))


def relu(x):
    return np.maximum(0, x)

def draw_chart(activation_fn):
    x = np.arange(-5.0, 5.0, 0.1)
    y = activation_fn(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

# draw_chart(step_function_using_numpy)
# draw_chart(sigmoid)
# draw_chart(relu)

x = np.array([[1, 2]])
w = np.array([[1, 3, 5], [2, 4, 6]])
y = np.dot(x, w)

print(y)

x = np.array([[1.0, 0.5]])

# 은닉 1층
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([[0.1, 0.2, 0.3]])
a1 = np.dot(x, w1) + b1
z1 = sigmoid(a1) # 1x3 => 새로운 입력층 (3개 노드)

# 은닉 2층
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([[0.1, 0.2]])
a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2) # 1x2 => 새로운 입력층 (2개 노드)

# 은닉 3층
w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([[0.1, 0.2]])
a3 = np.dot(z2, w3) + b3

def identity_function(a):
    print(a)

y = identity_function(a3) # 출력값을 식별함수를 이용해서 결과를 결정한다.

# 소프트맥스 함수
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

print(softmax(np.array([1010, 1000, 990])))
