# Definition for class VarClus

import pandas as pd
from sklearn.decomposition import PCA

from decomposition.base_class import BaseDecompositionClass


class Cluster:
    """
    A container that holds the following information
        - features in this cluster
        - first n PCA components and their corresponding eigenvalues
    """

    def __init__(self,
                 dataframe,
                 n_split=2,
                 features=None):

        self.features = features or dataframe.columns
        self.dataframe = dataframe[features]
        self.n_split = n_split

    def run_pca(self):
        self.pca = PCA(n_components=self.n_split)
        self.pca = self.pca.fit(self.dataframe)
        self.pca_features = []
        self.pca_corr = []

        for i in range(self.n_split):
            self.pca_features.append(self.dataframe.dot(self.pca.components_[i]))
            self.pca_corr.append(self.dataframe.corrwith(self.pca_features[i]))


class VarClus(BaseDecompositionClass):
    def __init__(self,
                 n_split=2):

        self.n_split = n_split

    @staticmethod
    def __reassign_one_feature(cluster_from, cluster_to, feature):
        cluster_from_new = Cluster(dataframe=cluster_from.drop(feature, axis=1),
                                   n_split=cluster_from.n_split)
        cluster_to_new = Cluster(dataframe=cluster_to.join(cluster_from.dataframe[feature]),
                                 n_split=cluster_to.n_split)

        for cluster in (cluster_from, cluster_from_new, cluster_to, cluster_to_new):
            if not getattr((cluster, 'pca', False)):
                cluster.run_pca()

        explained_variance_before_assignment = \
            cluster_from.pca.explained_variance_[0] + cluster_to.pca.explained_variance_[0]

        explained_variance_after_assignment = \
            cluster_from_new.pca.explained_variance_[0] + cluster_to_new.pca.explained_variance_[0]

        if explained_variance_after_assignment > explained_variance_before_assignment:
            return cluster_from_new, cluster_to_new, True
        else:
            return cluster_from, cluster_to, False

    @staticmethod
    def __reassign_features(child_clusters, max_tries=5):


    @staticmethod
    def __one_step_decompose(cluster):
        if not getattr(cluster, 'pca', False):
            cluster.run_pca()

        child_clusters = []

        corr_table = pd.concat(cluster.pca_corr, axis=1)
        corr_max = corr_table.max(axis=1)
        cluster_membership = corr_table.apply(lambda x: x == corr_max)

        for i in range(cluster.n_split):
            child_clusters.append(
                Cluster(dataframe=cluster.dataframe,
                        n_split=cluster.n_split,
                        features=[feature for (feature, condition)
                                  in cluster_membership[i].to_dict().items()
                                  if condition])
            )

        return child_clusters

    def fit(self, dataframe):
        """
        Decomposes given dataframe in an oblique hierarchical way.

        :param dataframe: feature space that needs to be decomposed
        """

        root_cluster = Cluster(dataframe,
                               self.n_split)

        root_cluster.run_pca()




