from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    
    mean_on_train = X_train.mean(axis=0)
    std_on_train = X_train.std(axis=0)

    #X_train_scaled = (X_train - mean_on_train) / std_on_train
    #X_test_scaled = (X_test - mean_on_train) / std_on_train
    X_train_scaled = X_train
    X_test_scaled = X_test
    
    mlp = MLPClassifier(random_state=0)
    mlp.fit(X_train_scaled, y_train)

    print("accuracy on training set: %f" % mlp.score(X_train_scaled, y_train))
    print("accuracy on test set: %f" % mlp.score(X_test_scaled, y_test))
