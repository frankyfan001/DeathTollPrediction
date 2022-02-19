import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)
        return self.w.copy()

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self, X, y, V):
        ''' YOUR CODE HERE '''
        self.w = solve(X.T@V@X, X.T@V@y)

    def predict(self, X):
        return X @ self.w

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w.flatten(), lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = 0.5*np.sum((X@w - y)**2)

        # Calculate the gradient value
        g = X.T@(X@w-y)

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        N, D = X.shape
        newColumn = np.ones((N, 1))
        Z = np.hstack((newColumn,X))
        self.v = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        ''' YOUR CODE HERE '''
        N, D = X.shape
        newColumn = np.ones((N, 1))
        Z = np.hstack((newColumn, X))
        return Z @ self.v

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        Z = self.__polyBasis(X)
        self.v = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        ''' YOUR CODE HERE '''
        Z = self.__polyBasis(X)
        return Z @ self.v

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        N, D = X.shape
        Z = np.zeros((N, self.p+1))
        x = X[:, 0]
        for p in range(self.p+1):
            Z[:, p] = x**p
        return Z
