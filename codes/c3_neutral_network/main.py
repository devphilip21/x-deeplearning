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
draw_chart(relu)
