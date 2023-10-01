import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC

if __name__ == "__main__":
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
            random_state=0)

    svm = SVC(C=100)
    svm.fit(X_train, y_train)
    print(svm.score(X_test, y_test))

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm.fit(X_train_scaled, y_train)
    print(svm.score(X_test_scaled, y_test))

