import numpy as np
from ml.naive_bayes_1 import NaiveBayes

N = 1000
x_1 = np.random.standard_normal((N, 2))
for i in range( x_1.shape[0] ):
    x_1[i][0] = 3 * x_1[i][0]
    x_1[i][1] = 0.5 * x_1[i][0] * x_1[i][0]
    x_1[i][1] = x_1[i][1] + 2.0 * np.random.standard_normal(1)
x_2 = np.random.standard_normal((N, 2))
for i in range( x_2.shape[0] ):
    x_2[i][0] = x_2[i][0]
    x_2[i][1] = x_2[i][1] + 10.0

x = np.vstack( (x_1, x_2) )
y = np.zeros((x.shape[0],))
for i in range(y.shape[0]):
    if i < y.shape[0] // 2:
        y[i] = 1
    else:
        y[i] = -1

nb = NaiveBayes()
nb.fit(x, y)

##################################
import matplotlib.pyplot as plt

plt.scatter(x[y == -1,0], x[y == -1,1], s=1)
plt.scatter(x[y == 1,0], x[y == 1,1], s=1)

x = 4*np.random.standard_normal((N, 2))
x[:, 1] = x[:, 1] + 7.0
y = nb.predict(x)
plt.scatter(x[y == -1,0], x[y == -1,1], s=1)
plt.scatter(x[y == 1,0], x[y == 1,1], s=1)

plt.show()