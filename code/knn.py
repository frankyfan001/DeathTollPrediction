"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        # Compute cosine_distance distances between X and Xtest
        dist2 = self.cosine_distance(X, Xtest)

        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            # sort the distances to other points
            inds = np.argsort(dist2[:, i])

            # compute mode of k closest training pts
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat

    def cosine_distance(self, X1, X2):
        "Insert your code here"

        # We will first compute cosine similarity, then transform it to cosine distance.
        # cosine_similarity(a, b) = (a . b) / (|a| * |b|)

        # numerator:    a . b
        numeratorMatrix = X1 @ (X2.T)

        # denominator:  |a| * |b|
        norms1 = np.linalg.norm(X1, axis=1).reshape((X1.shape[0], 1))
        norms2 = np.linalg.norm(X2, axis=1).reshape((X2.shape[0], 1))
        denominatorMatrix = norms1 @ (norms2.T)

        # if there is any zero row in X1 or X2, set the distance between any zero row in X1 to all the rows X2
        # to zero or vice verca. So we can play a trick here by setting denominator to inf in the case above.
        denominatorMatrix = np.where(denominatorMatrix == 0, np.inf, denominatorMatrix)

        # transform cosine similarity to cosine distance
        cosineSimilarity = (numeratorMatrix / denominatorMatrix)
        cosineDistance = 1 - cosineSimilarity
        return cosineDistance



        '''
        if there is any zero row in X1 or X2, set the distance between any zero row in X1 to all the rows X2 to zero or vice verca.
        '''
