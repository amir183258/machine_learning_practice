import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

if __name__ == "__main__":
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)

    print("accuracy on training set: %f" % tree.score(X_train, y_train))
    print("accuracy on training set: %f" % tree.score(X_test, y_test))

    export_graphviz(tree, out_file="mytree.dot", class_names=["malignant", "benign"],
            feature_names=cancer.feature_names, impurity=False, filled=True)

    with open("mytree.dot") as f:
        dot_graph = f.read()

    src = graphviz.Source(dot_graph)
    src.render('./text.pdf')
    
    """
    tree = DecisionTreeClassifier(random_state=0, max_depth=4)
    tree.fit(X_train, y_train)

    print("accuracy on training set: %f" % tree.score(X_train, y_train))
    print("accuracy on training set: %f" % tree.score(X_test, y_test))
    """
