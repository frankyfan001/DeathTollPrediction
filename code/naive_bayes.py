import matplotlib.pyplot as plt
import numpy as np

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        # Important: First clean dataset by removing its duplicate data. Otherwise, the duplicate data will affect
        # the expected shape of normal distribution (mean stdDev), thus increases training and testing errors.
        Xy = np.hstack((X, y.reshape((len(y), 1))))
        Xy_unique = np.unique(Xy, axis=0)
        N, D = Xy_unique.shape
        X = Xy_unique[:, 0:D-1]
        y = Xy_unique[:, D-1].astype(y.dtype)

        # Compute the probability of each class i.e p(y==c).
        N, D = X.shape
        counts = np.bincount(y)
        p_y = counts / N

        # Compute the number of class labels.
        C = len(counts)

        # Let x_dc represent data entry from X with feature d and class c.
        # Compute the mean and stdDev for each x_dc.
        meanMatrix = np.zeros((D, C), float)
        stdDevMatrix = np.zeros((D, C), float)
        for d in range(D):
            for c in range(C):
                x_dc = X[:, d][(y == c)]
                meanMatrix[d, c] = np.mean(x_dc)
                stdDevMatrix[d, c] = np.std(x_dc)

        self.c = C
        self.p_y = p_y
        self.meanMatrix = meanMatrix
        self.stdDevMatrix = stdDevMatrix

    def predict(self, X):
        C = self.c
        p_y = self.p_y
        mean = self.meanMatrix
        stdDev = self.stdDevMatrix

        N, D = X.shape
        y_pred = np.zeros(N, int)
        for i in range(N):
            # make predictions p(y) in log space.
            log_p_y = np.log(p_y)
            # make predictions p(x|y) in log space.
            x = np.array([X[i],]*C).T
            log_p_xy = np.sum((-0.5 * (((x - mean) / stdDev) ** 2) - np.log(stdDev * np.sqrt(2 * np.pi))), axis=0)
            # p(x|y) * p(y) in log space
            result = log_p_xy + log_p_y

            y_pred[i] = np.argmax(result)

        return y_pred
