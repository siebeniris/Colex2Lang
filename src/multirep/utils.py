import numpy as np
from gensim.models import keyedvectors
from matplotlib import pyplot as plt

from sklearn.metrics import pairwise_distances, silhouette_score

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
from scipy.spatial.distance import pdist

PATH_FIGS = "./figs_obj.v3/"


def load_features(dataset, ng_model="node2vec", languages=None):
    feature_vector_path = f"data/language_embeddings/{dataset}_{ng_model}_embeddings"

    X_total = []
    feature_vectors = keyedvectors.load_word2vec_format(feature_vector_path, binary=False)
    feature_keys = list(feature_vectors.key_to_index.keys())

    if languages is not None:
        languages_ = set(languages).intersection(set(feature_keys))
        label_total = sorted(list(languages_))
    else:
        label_total = sorted(feature_keys)

    for l in label_total:
        X_total.append(feature_vectors[l])

    return np.array(X_total), label_total


def inertia_score(X, a, metric="euclidean"):
    '''
    a: assignments (predictions)
    X: dataset
    '''
    W = [np.mean(pairwise_distances(X[a == c, :], metric=metric)) for c in np.unique(a)]
    return np.mean(W)


def linkage_matrix(X, method="ward", metric="euclidean"):
    y = pdist(X, metric=metric)
    if method == "ward":
        # y = pdist(X, metric = metric)
        Z = ward(y)
    else:
        # Z = linkage(X, method = method, metric = metric)
        Z = linkage(y, method=method, metric=metric)
    return Z, y


new_colors = ['#1f77b4', '#17becf', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#ff7f0e']


def save_numc_and_dend_plot(Z, langs_names, maxclusters, X_inertia, X_silhouette, name, path, color_th=0, ang_rot=90,
                            orient="top"):
    if path is None:
        path = PATH_FIGS
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 2), gridspec_kw={'width_ratios': [1, 1, 3]})

    x_items = min(20, maxclusters + 1)
    if x_items < len(X_inertia):
        X_inertia = X_inertia[:x_items - 2]
        X_silhouette = X_silhouette[:x_items - 2]
    # First subplot: Elbow method (Inertia)
    a0.title.set_text("Elbow method")
    a0.grid(True, axis='x', linewidth=0.5)
    a0.plot(range(2, x_items), X_inertia, '-')  # , label='data')
    a0.set_xlabel('# clusters')
    # a0.set_ylabel('Inertia score')
    a0.set_xticks(np.arange(2, x_items, step=2))
    a0.tick_params(axis='x', which='minor', labelsize=2)

    # Second subplot: Silhouette analysis
    a1.title.set_text("Silhouette analysis")
    # a1.yaxis.tick_right()
    # a1.yaxis.set_label_position("right")
    a1.grid(True, axis='x', linewidth=0.5)
    a1.plot(range(2, x_items), X_silhouette, '-')
    # a1.set_ylabel('Silhouetthe score')
    a1.set_xlabel('# clusters')
    a1.set_xticks(np.arange(2, x_items, step=2))

    # Third subplot: Dendrogram
    # analyse color_th:
    if color_th == 0: color_th = np.median(Z[:, 2])

    a2.title.set_text("Dendrogram: " + name)
    a2.yaxis.tick_right()
    a2.yaxis.set_label_position("right")
    a2.grid(False)
    hierarchy.set_link_color_palette(new_colors[0:7] + new_colors[8:])
    hierarchy.dendrogram(Z=Z,
                         orientation=orient,
                         labels=langs_names,
                         distance_sort='descending',
                         color_threshold=color_th,
                         show_leaf_counts=True,
                         leaf_rotation=ang_rot,
                         above_threshold_color=new_colors[7]
                         )
    f.savefig(path + name + ".pdf", bbox_inches='tight', transparent=True)
    plt.close()


if __name__ == '__main__':
    X_total, label_total = load_features("wn_concept")
    print(X_total.shape)
    print(label_total)
