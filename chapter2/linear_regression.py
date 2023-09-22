import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


if __name__ == "__main__":
    X, y = mglearn.datasets.make_wave(n_samples=60)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    lr = LinearRegression().fit(X_train, y_train)

    print("lr.coef_: %s" % lr.coef_)
    print("lr.intercept_: %s" % lr.intercept_)

    print("training set score: %f" % lr.score(X_train, y_train))
    print("test set score: %f" % lr.score(X_test, y_test))
