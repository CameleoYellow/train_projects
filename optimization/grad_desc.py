#minimizing scalar func of vector argument

import numpy as np
import scipy as sp
from copy import deepcopy

def func(x):
    y = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i == j:
                y += x[i] ** 2
            else:
                y += 2 * x[i] * x[j]
    return y

def fGrad(x, f, step=np.sqrt(np.finfo(float).eps)):
    grad = np.zeros(x.shape)
    for i in range(x.shape[0]):
        partDx = deepcopy(x)
        partDx[i] = x[i] + step
        grad[i] = f(partDx) - f(x)
        grad[i] = grad[i] / step
    return grad

def gradStep(x,f, lr=1.0):
    grad = fGrad(x, f)
    nextX = x - lr * x * (grad / np.linalg.norm(grad))
    return nextX


print(np.finfo(float).eps)
print(np.sqrt(np.finfo(float).eps))
eps = 0.001
x = np.random.random(5)
prevX = np.random.rand(x.shape[0])
while abs(func(x) - func(prevX)) > eps:
    print("x = {} | y = {}".format(x, func(x)))
    prevX = x
    x = gradStep(x, func)

