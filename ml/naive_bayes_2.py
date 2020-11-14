
import numpy as np

def gaussPDF(x, u, covMat):
    n = x.shape[0]
    pdf = 1 / np.sqrt( (2 * np.pi)**n * np.linalg.det(covMat))
    invCovMat = np.linalg.inv(covMat)
    #vec = x - u
    #vec = np.reshape(vec, (2, 1))
    pdf = pdf * np.exp( - np.matmul( np.matmul((x - u).T, invCovMat), (x - u)) )
    return pdf


class NaiveBayes():
    def __init__(self):

        self.nClasses = 0
        self.classes = []
        self.py = {}
        self.u = {}
        self.s = {}

        pass

    def fit(self, x, y):
        uniqLabels = list(sorted(list(np.unique(y))))
        self.classes = uniqLabels
        self.nClasses = len(uniqLabels)
        for uLabel in uniqLabels:
            self.py[uLabel] = y[y == uLabel].shape[0] / y.shape[0]
            self.s[uLabel] = np.cov(x[y == uLabel], rowvar=False)
            self.u[uLabel] = []
            for i in range(self.nClasses):
                self.u[uLabel].append(np.mean(x[y == uLabel, i]))
            self.u[uLabel] = np.array(self.u[uLabel])
        pass

    def get_estimates(self, x):
        """ Class probability estimation """
        estimates = np.zeros((x.shape[0], self.nClasses))
        for i in range( x.shape[0] ):
            for j in range( self.nClasses ):
                currentClass = self.classes[j]
                estimates[i][j] = np.log( self.py[currentClass] )
                estimates[i][j] += np.log( gaussPDF(x[i], self.u[currentClass], self.s[currentClass]) )
        return estimates

    def predict(self, x):

        estimates = self.get_estimates(x)
        classes = []
        for i in range( estimates.shape[0] ):
            classes.append( int(self.classes[np.argmax(estimates[i])]) )
        classes = np.array(classes)

        return classes

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