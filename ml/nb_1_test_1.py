import numpy as np
from ml.naive_bayes_1 import NaiveBayes


N = 1000
x = np.random.standard_normal((N, 2))

x = np.vstack( (2 * x - 10.0, 2 * x + 10.0) )
y = np.zeros((x.shape[0],))
for i in range(y.shape[0]):
    if i < y.shape[0] // 2:
        y[i] = 1
    else:
        y[i] = -1

nb = NaiveBayes()
nb.fit(x, y)
nb.get_estimates(x)
predicts = nb.predict(x)

##################################
import matplotlib.pyplot as plt

plt.scatter(x[:,0], x[:,1], s=1)

x = 4*np.random.standard_normal((N, 2))
y = nb.predict(x)
tmp = [y == -1]
plt.scatter(x[y == -1,0], x[y == -1,1], s=1)
plt.scatter(x[y == 1,0], x[y == 1,1], s=1)

plt.show()