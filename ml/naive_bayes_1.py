
import numpy as np

def gaussPDF(x, u, s):
    pdf = 1 / (s * np.sqrt(2 * np.pi))
    pdf = pdf * np.exp( - (x - u)**2 / (2 * s**2) )
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
            self.u[uLabel] = []
            self.s[uLabel] = []
            for i in range(self.nClasses):
                self.u[uLabel].append(np.mean(x[y == uLabel, i]))
                self.s[uLabel].append(np.std(x[y == uLabel, i]))
        pass

    def get_estimates(self, x):
        """ Class probability estimation """
        estimates = np.zeros((x.shape[0], self.nClasses))
        for i in range( x.shape[0] ):
            for j in range( self.nClasses ):
                currentClass = self.classes[j]
                estimates[i][j] = np.log( self.py[currentClass] )
                for k in range(x.shape[1]):
                    estimates[i][j] += np.log( gaussPDF(x[i][k], self.u[currentClass][k], self.s[currentClass][k]) )
        return estimates

    def predict(self, x):

        estimates = self.get_estimates(x)
        classes = []
        for i in range( estimates.shape[0] ):
            classes.append( int(self.classes[np.argmax(estimates[i])]) )
        classes = np.array(classes)

        return classes