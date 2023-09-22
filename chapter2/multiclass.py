import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    X, y = make_blobs(random_state=42)

    linear_svm = LinearSVC().fit(X, y)
    
    print(linear_svm.coef_.shape)
    print(linear_svm.intercept_.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm3)
    line = np.linspace(-15, 15)
    for coef, intercept in zip(linear_svm.coef_, linear_svm.intercept_):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1])
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)


    plt.show()
    

    
