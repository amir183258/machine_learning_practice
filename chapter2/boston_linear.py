import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np

if __name__ == "__main__":
    X, y = mglearn.datasets.load_extended_boston()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LinearRegression().fit(X_train, y_train)

    print("Normal Regression: ")
    print("training set score: %f" % lr.score(X_train, y_train))
    print("test set score: %f" % lr.score(X_test, y_test))

    # Ridge
    ridge = Ridge().fit(X_train, y_train)
    print("Ridge Regression: ")
    print("training set score: %f" % ridge.score(X_train, y_train))
    print("test set score: %f" % ridge.score(X_test, y_test))

    # Ridge 10
    ridge10 = Ridge(alpha=10).fit(X_train, y_train)
    print("Ridge Regression 10: ")
    print("training set score: %f" % ridge10.score(X_train, y_train))
    print("test set score: %f" % ridge10.score(X_test, y_test))

    # Lasso
    lasso = Lasso().fit(X_train, y_train)
    print("Lasso Regression: ")
    print("training set score: %f" % lasso.score(X_train, y_train))
    print("test set score: %f" % lasso.score(X_test, y_test))
    print("number of features used: %d" % np.sum(lasso.coef_ != 0))

    # Lasso 0.01
    lasso001 = Lasso(alpha=0.01).fit(X_train, y_train)
    print("Lasso Regression: ")
    print("training set score: %f" % lasso001.score(X_train, y_train))
    print("test set score: %f" % lasso001.score(X_test, y_test))
    print("number of features used: %d" % np.sum(lasso001.coef_ != 0))
