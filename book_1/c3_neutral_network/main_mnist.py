import sys
sys.path.append('book_1')
import numpy as np
from dataset.mnist import load_mnist
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, # 배열로 할건지
        normalize=True, # 정규화 할건지 (0~1로)
        one_hot_label=False, # 01000 과 같이 t를 정답만 1로, 나머지는 0으로 표현할건지
    )

    return x_test, t_test

def init_network():
    with open('book_1/dataset/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def calc_accuracy_rate():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스
        if p == t[i]:
            accuracy_cnt += 1

    print(float(accuracy_cnt) / len(x))

def calc_accuracy_rate_batch():
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print(float(accuracy_cnt) / len(x))

calc_accuracy_rate()
calc_accuracy_rate_batch()
