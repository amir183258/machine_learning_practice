import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, random_state=0)

    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)

    print("accuracy on training set: %f" % gbrt.score(X_train, y_train))
    print("accuracy on training set: %f" % gbrt.score(X_test, y_test))

    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)

    print("accuracy on training set: %f" % gbrt.score(X_train, y_train))
    print("accuracy on training set: %f" % gbrt.score(X_test, y_test))

    gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt.fit(X_train, y_train)

    print("accuracy on training set: %f" % gbrt.score(X_train, y_train))
    print("accuracy on training set: %f" % gbrt.score(X_test, y_test))
