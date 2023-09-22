import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

if __name__ == "__main__":
    X, y = mglearn.datasets.make_wave(n_samples=40)

    mglearn.plots.plot_knn_regression(n_neighbors=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = KNeighborsRegressor(n_neighbors=3)

    reg.fit(X_train, y_train)

    KNeighborsRegressor(algorithm="auto", leaf_size=30, metric="minkowski",
            metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights="uniform")

    print(reg.score(X_test, y_test))

    print(len(X))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    print(line.shape)

    plt.suptitle("nearest_neighbor_regression")
    for n_neighbors, ax in zip([1, 3, 9], axes):
        reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
        ax.plot(X, y, 'o')
        ax.plot(X, -3 * np.ones(len(X)), 'o')
        ax.plot(line, reg.predict(line))
        ax.set_title("%d neighbor(s)" % n_neighbors)
    plt.show()



