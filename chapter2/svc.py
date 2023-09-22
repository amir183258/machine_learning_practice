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
    X, y = mglearn.tools.make_handcrafted_dataset()
    svm = SVC(kernel="rbf", C=10, gamma=0.1).fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, eps=0.5)

    plt.scatter(X[:, 0], X[:, 1], s=60, c=y, cmap=mglearn.cm2)

    sv = svm.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=200, facecolors="none", zorder=10, linewidth=3, color="black")

    plt.show()
