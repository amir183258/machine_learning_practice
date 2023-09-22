import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":
    X, y = mglearn.datasets.make_forge()

    #plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
    #mglearn.plots.plot_knn_classification(n_neighbors=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X_train, y_train)

    KNeighborsClassifier(algorithm="auto", leaf_size=30, metric="minkowski", metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights="uniform")

    y_pred = clf.predict(X_test)

    score = clf.score(X_test, y_test)

    print(score)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
        ax.set_title("%d neighbor(s)" % n_neighbors)

    print(X)

    plt.show()
