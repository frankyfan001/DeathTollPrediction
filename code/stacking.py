import numpy as np
from scipy import stats

from random_forest import RandomForest, DecisionTree
from knn import KNN
from naive_bayes import NaiveBayes

class Stacking():

    def __init__(self):
        # classifiers
        self.randomForest = RandomForest(max_depth=np.inf, num_trees=15)
        self.knn = KNN(k=3)
        self.naiveBayes = NaiveBayes()

        # meta-classifier
        self.decisionTree = DecisionTree(max_depth=np.inf)  # stump_class=DecisionStumpErrorRate by default.

    def fit(self, X, y):
        self.randomForest.fit(X, y)
        self.knn.fit(X, y)
        self.naiveBayes.fit(X, y)

    def predict(self, X):
        pred0 = self.randomForest.predict(X)
        pred1 = self.knn.predict(X)
        pred2 = self.naiveBayes.predict(X)

        # Use predictions of the classifiers to form X y for meta-classifier.
        X = np.vstack((pred0, pred1, pred2)).T
        y = stats.mode(X, axis=1)[0].flatten()

        # Train meta-classifier and return predictions.
        self.decisionTree.fit(X, y)
        return self.decisionTree.predict(X)
