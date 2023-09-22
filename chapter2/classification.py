import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":
    """
    X, y = mglearn.datasets.make_forge()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    plt.suptitle("linear_classifiers")

    for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
        clf = model.fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
        ax.set_title("%s" % clf.__class__.__name__)
    """

    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    logisticregression = LogisticRegression().fit(X_train, y_train)

    print("training set score: %f" % logisticregression.score(X_train, y_train))
    print("test set score: %f" % logisticregression.score(X_test, y_test))

    # C = 100
    logisticregression100 = LogisticRegression(C=100).fit(X_train, y_train)

    print("training set score: %f" % logisticregression100.score(X_train, y_train))
    print("test set score: %f" % logisticregression100.score(X_test, y_test))


    
    

    
