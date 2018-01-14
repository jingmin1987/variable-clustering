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

    def fit(self, dataframe):
        """
        Decomposes given dataframe in an oblique hierarchical way.

        :param dataframe: feature space that needs to be decomposed
        """

