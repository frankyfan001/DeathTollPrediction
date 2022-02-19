
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils
from random_forest import RandomForest
from knn import KNN
from naive_bayes import NaiveBayes
from stacking import Stacking

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # load X, y
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_train.csv'))
        Xy_train = df.values
        names_train = df.columns.values
        N, D = Xy_train.shape
        X= Xy_train[:, 0:D-1]                     # test only first 100 rows
        y = Xy_train[:, D-1].astype(int)          # test only first 100 rows

        # load X_test, y_test
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_test.csv'))
        Xy_test = df.values
        names_test = df.columns.values
        N, D = Xy_test.shape
        X_test = Xy_test[:, 0:D-1]
        y_test = Xy_test[:, D-1].astype(int)

        # create model
        # model = RandomForestClassifier(max_depth=None, n_estimators=15)
        model = RandomForest(max_depth=np.inf, num_trees=15)

        # evaluate model
        print("Random Forest")
        print("Inner Kmeans's k=15")
        t0 = datetime.datetime.now()
        utils.evaluate_model(model, X, y, X_test, y_test)
        t1 = datetime.datetime.now()
        print("Runtime:", t1 - t0)

    elif question == "1.2":
        # load X, y
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_train.csv'))
        Xy_train = df.values
        names_train = df.columns.values
        N, D = Xy_train.shape
        X= Xy_train[:, 0:D-1]                     # test only first 100 rows
        y = Xy_train[:, D-1].astype(int)          # test only first 100 rows

        # load X_test, y_test
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_test.csv'))
        Xy_test = df.values
        names_test = df.columns.values
        N, D = Xy_test.shape
        X_test = Xy_test[:, 0:D-1]
        y_test = Xy_test[:, D-1].astype(int)

        # create model
        # model = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        model = KNN(k=3)

        # evaluate model
        print("KNN")
        print("k=3")
        t0 = datetime.datetime.now()
        utils.evaluate_model(model, X, y, X_test, y_test)
        t1 = datetime.datetime.now()
        print("Runtime:", t1 - t0)

    elif question == "1.3":
        # load X, y
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_train.csv'))
        Xy_train = df.values
        names_train = df.columns.values
        N, D = Xy_train.shape
        X= Xy_train[:, 0:D-1]                     # test only first 100 rows
        y = Xy_train[:, D-1].astype(int)          # test only first 100 rows

        # load X_test, y_test
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_test.csv'))
        Xy_test = df.values
        names_test = df.columns.values
        N, D = Xy_test.shape
        X_test = Xy_test[:, 0:D-1]
        y_test = Xy_test[:, D-1].astype(int)

        # create model
        # model = BernoulliNB()
        model = NaiveBayes()

        # evaluate model
        print("Naive Bayes")
        t0 = datetime.datetime.now()
        utils.evaluate_model(model, X, y, X_test, y_test)
        t1 = datetime.datetime.now()
        print("Runtime:", t1 - t0)

    elif question == "1.4":
        # load X, y
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_train.csv'))
        Xy_train = df.values
        names_train = df.columns.values
        N, D = Xy_train.shape
        X= Xy_train[:, 0:D-1]                     # test only first 100 rows
        y = Xy_train[:, D-1].astype(int)          # test only first 100 rows

        # load X_test, y_test
        df = pd.read_csv(os.path.join('..', 'data', 'wordvec_test.csv'))
        Xy_test = df.values
        names_test = df.columns.values
        N, D = Xy_test.shape
        X_test = Xy_test[:, 0:D-1]
        y_test = Xy_test[:, D-1].astype(int)

        # Get average errors.
        n = 100
        result = np.zeros((n, 2), float)
        for i in range(0, n):
            # create model
            model = Stacking()

            # evaluate model
            print()
            print("Stacking%d" %i)
            t0 = datetime.datetime.now()
            result[i] = utils.evaluate_model(model, X, y, X_test, y_test)
            t1 = datetime.datetime.now()
            print("Runtime:", t1 - t0)

        result = np.sum(result, axis=0)/n
        print()
        print("Average Training error: %.3f" % result[0])
        print("Average Testing error: %.3f" % result[1])

    if question == "2":

        ####################################################################################################
        # In order to make this code concise and clear, we have attached the fig.2 "Constructing X, Y, W"
        # in the report instead of tedious inline comments. Please refer to the figure while reading through
        # our implementation. It explains the details in a single picture and demonstrate how to construct
        # X Y W in a Linear Autoregressive Model.
        ####################################################################################################

        # Helper function to process dataset to build a time series raw data.
        def processDataset(datasetName, countries, features):
            with open(os.path.join("..", "data", datasetName), "rb") as f:
                dataset = pd.read_csv(f, header=0)
            C = len(countries)
            F = len(features)
            result = None
            for c in range(C):
                country = dataset.loc[dataset['country_id'] == countries[c]]
                for f in range(F):
                    row = country[features[f]].to_numpy().copy()
                    result = row if result is None else np.vstack((result, row))
            return result.T

        ###################################################################### 0. Adjust hyper-parameters.

        countries = np.array(["CA", "UK"])          # Parameter countries: the selected countries in X.
        features = np.array(["deaths", "cases"])    # Parameter features: the selected features of each country in X.
        K = 15                                      # Parameter K: the number of previous days in a feature vector.
        P = 5                                       # Parameter P: the number of days to predict.

        ###################################################################### 1. Build time series rawData.

        datasetName = "phase2_training_data.csv"
        rawData = processDataset(datasetName, countries, features)
        rawData = rawData[:len(rawData)-5]  # For test only (comment out).

        ###################################################################### 2. Build X Y (Augoregressive Model).

        ONE = 1
        C = len(countries)
        F = len(features)
        T = rawData.shape[0]
        N = T-K
        D = K*C*F + ONE

        X = np.zeros((N, D), float)
        Y = np.zeros((N, C*F), float)

        featureVector = np.ones(D, float)
        p0 = (K-1)*C*F
        p1 = C*F
        p2 = K*C*F
        for i in range(N):
            if i == 0:
                featureVector[:D-ONE] = rawData[:K].flatten()
            else:
                featureVector[:p0] = featureVector[p1:p2]
                featureVector[p0:p2] = Y[i-1]
            X[i] = featureVector
            Y[i] = rawData[K+i]

        ###################################################################### 3. Train to build W.

        W = np.zeros((D, C * F), float)
        model = linear_model.LeastSquares()
        for i in range(C*F):
            W[:, i] = model.fit(X, Y[:, i])

        ###################################################################### 4. Predict to build Y_pred.

        Y_pred = np.zeros((P, C*F), float)
        for i in range(P):
            featureVector[:p0] = featureVector[p1:p2]
            if i == 0:
                featureVector[p0:p2] = Y[N-1]
            else:
                featureVector[p0:p2] = Y_pred[i-1]
            Y_pred[i] = featureVector.T@W

        ###################################################################### 5. Compute error.

        y_pred = Y_pred[:, 0]
        y_test = np.array([9794, 9829, 9862, 9888, 9922])   # For test only (comment out).
        root_mean_squared_error = np.sqrt(np.sum((y_pred - y_test)**2) / len(y_pred))
        print(root_mean_squared_error)

    else:
        print("Unknown question: %s" % question)
