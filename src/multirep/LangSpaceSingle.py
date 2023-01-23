import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statistics import mean, median, stdev
from scipy.stats import median_absolute_deviation
from random import shuffle

from src.multirep.utils import load_features
from src.multirep.ClusterData import ClusterData
from src.multirep.langcodes import Baseline_Langs


class LangSpaceSingle:
    def __init__(self, dataset, languages, ng_model="node2vec", attributes=None, X_total=None, label_total=None,
                 svd_transform=False):
        # already learned language node embeddings, no svd transformation needed.
        self.feature_name = dataset
        self.attributes = attributes

        if X_total is not None and label_total is not None:
            self.X_total = X_total
            self.label_total = label_total
        else:
            self.X_total, self.label_total = load_features(dataset, ng_model, languages)

        self.num_langs = len(self.label_total)

    def print_status(self):
        print(f"feature name : {self.feature_name}")
        print(f"nr of langauges: {self.num_langs}, top 10 {self.label_total}")
        print(f"feature vector shape {self.X_total.shape}")

    def getLabels(self):
        return self.label_total

    def getFeatureName(self):
        """feature name: similarity, phonological, syntactic, inventory,
        geographic, genetic, colex,
        featural
        """
        return self.feature_name

    def compute_single_cluster(self, method="average", metric="cosine"):
        cluster = ClusterData(self.X_total, self.label_total, max_clusters=self.num_langs - 2, name=self.feature_name,
                              metric=metric, method=method)

        return cluster


def create_single_phy_trees():
    for dataset in ["wn", "wn_concept", "clics"]:
        feature_name = dataset

        languages = list(Baseline_Langs.keys())
        lang_space = LangSpaceSingle(feature_name, languages)

        lang_space.print_status()
        cluster = lang_space.compute_single_cluster()
        cluster.plot_cluster_analysis(name=f"{feature_name}-baseline", path="data/phytree/SingleSpace/")


if __name__ == '__main__':
    create_single_phy_trees()