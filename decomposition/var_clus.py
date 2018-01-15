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


class VarClus(BaseDecompositionClass):
    def __init__(self,
                 n_split=2):

        self.n_split = n_split

    @staticmethod
    def __reassign_one_feature():
        pass

    @staticmethod
    def __one_step_decompose(cluster):
        if not getattr(cluster, 'pca', False):
            cluster.run_pca()

        pca_features = []
        pca_corr = []
        child_clusters = []

        for i in range(cluster.n_split):
            pca_features.append(cluster.dataframe.dot(cluster.pca.components_[i]))
            pca_corr.append(cluster.dataframe.corrwith(pca_features[i]))

        corr_table = pd.concat(pca_corr, axis=1)
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




