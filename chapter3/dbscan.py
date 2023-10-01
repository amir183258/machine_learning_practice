import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import mglearn
from sklearn.datasets import fetch_lfw_people
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    counts = np.bincount(people.target)

    mask = np.zeros(people.target.shape, dtype=bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]

    pca = PCA(n_components=100, whiten=True)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    n_clusters = 10
    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels_km = km.fit_predict(X_pca)
    print("cluster sizes k-Means %s" % np.bincount(labels_km))

    fig, axes = plt.subplots(2, 5, subplot_kw={"xticks": (), "yticks": ()}, figsize=(12, 4))
    for center, ax in zip(km.cluster_centers_, axes.ravel()):
        ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
    plt.show()







