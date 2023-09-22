from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    min_on_training = X_train.min(axis=0)
    range_on_training = (X_train - min_on_training).max(axis=0)

    X_train_scaled = (X_train - min_on_training) / range_on_training

    X_test_scaled = (X_test - min_on_training) / range_on_training

    svc = SVC(C=1000)
    svc.fit(X_train, y_train)

    print("accuracy on training set: %f" % svc.score(X_train, y_train))
    print("accuracy on test set: %f" % svc.score(X_test, y_test))

