from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs, make_circles

from sklearn.ensemble import GradientBoostingClassifier

if __name__ == "__main__":
    X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
    y_named = np.array(["blue", "red"])[y]

    X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X,\
            y_named, y, random_state = 0)

    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train_named)

    GradientBoostingClassifier(init=None, learning_rate=0.1, loss="deviance",
            max_depth=3, max_features=None, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100,
            random_state=0, subsample=1.0, verbose=0,
            warm_start=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
    score_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4, cm="bwr")
    
    for ax in axes:
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=mglearn.cm2, s=60, marker='^')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=mglearn.cm2, s=60)
    plt.colorbar(score_image, ax=axes.tolist())
    plt.show()



