from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2

    X_new = np.hstack([X, X[:, 1:] ** 2])

    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    xx = np.linspace(X_new[:, 0].min(), X_new[:, 0].max(), 50)
    yy = np.linspace(X_new[:, 1].min(), X_new[:, 1].max(), 50)

    XX, YY = np.meshgrid(xx, yy)
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]


    fig = plt.figure()
    ax = Axes3D(fig)

    ax = fig.add_subplot(111, projection='3d', facecolor='#FFAAAA')
    ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=y, cmap=mglearn.cm2, s=60)
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    ax.set_xlabel("feature1")
    ax.set_ylabel("feature2")
    ax.set_zlabel("feature1 ** 2")

    plt.show()

