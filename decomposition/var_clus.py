# Definition for class VarClus
# TODO: add log function, and docstrings, and cleanup the prints


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from decomposition.base_class import BaseDecompositionClass


class Cluster:
    """
    A tree-node type container that holds the following information
        - features in this cluster
        - first n PCA components and their corresponding eigenvalues
    """

    def __init__(self, dataframe, n_split=2, features=None, parents=None, children=None, name=None):

        # Using dataframe.columns will generate an index-list which is not convertible to set
        self.features = features or list(dataframe)
        self.dataframe = dataframe[self.features]
        self.n_split = n_split
        self.parents = parents or []
        self.children = children or []
        self.name = name or ''

        self.input_check()

    def run_pca(self):
        self.pca = PCA(n_components=self.n_split)
        self.pca = self.pca.fit(self.dataframe)
        self.pca_features = []
        self.pca_corr = []

        for i in range(self.n_split):
            self.pca_features.append(self.dataframe.dot(self.pca.components_[i]))
            self.pca_corr.append(self.dataframe.corrwith(self.pca_features[i]))

    def input_check(self):
        if type(self.features) is not list:
            self.features = [self.features]

        if len(self.features) < self.n_split:
            print('number of features is smaller than n_split')
            self.n_split = len(self.features)

    def return_all_leaves(self):
        if self.children == []:
            return self

        return [child.return_all_leaves() for child in self.children]

    def __key(self):
        return (tuple(self.features), self.dataframe.shape)

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())


class VarClus(BaseDecompositionClass):
    def __init__(self, n_split=2, max_eigenvalue=1, max_tries=None):
        self.n_split = n_split
        self.max_eigenvalue = max_eigenvalue
        self.max_tries = max_tries

    @staticmethod
    def reassign_one_feature_pca(cluster_from, cluster_to, feature, other_clusters=None):
        if not (feature in cluster_from.features):
            return cluster_from, cluster_to

        if feature in cluster_to.features:
            print('feature {} is already in cluster_to'.format(feature))
            return cluster_from, cluster_to

        print('assessing feature {}'.format(feature))

        other_clusters = other_clusters or []

        cluster_from_new_df = cluster_from.dataframe.drop(feature, axis=1)
        cluster_to_new_df = cluster_to.dataframe.join(cluster_from.dataframe[feature])
        cluster_from_new = Cluster(dataframe=cluster_from_new_df,
                                   n_split=cluster_from.n_split,
                                   parents=cluster_from.parents)
        cluster_to_new = Cluster(dataframe=cluster_to_new_df,
                                 n_split=cluster_to.n_split,
                                 parents=cluster_to.parents)

        if len(cluster_from.features + cluster_to.features) != \
            len(cluster_from_new.features + cluster_to_new.features):
            missing_feature = set(cluster_from.features + cluster_to.features) - \
                set(cluster_from_new.features + cluster_to_new.features)
            print('feature missing....the missing feature is...{}').format(missing_feature)

        for cluster in [cluster_from, cluster_from_new, cluster_to, cluster_to_new]:
            if not getattr(cluster, 'pca', False):
                cluster.run_pca()

        explained_variance_before_assignment = np.sum(
            [cluster.pca.explained_variance_[0] for cluster in ([cluster_from, cluster_to]
                                                                + other_clusters)],
        )

        explained_variance_after_assignment = np.sum(
            [cluster.pca.explained_variance_[0] for cluster in ([cluster_from_new, cluster_to_new]
                                                                + other_clusters)],
        )

        print('current EV is {0}, new EV is {1}'.format(explained_variance_before_assignment,
                                                        explained_variance_after_assignment))

        if explained_variance_after_assignment > explained_variance_before_assignment:
            return cluster_from_new, cluster_to_new, True
        else:
            return cluster_from, cluster_to, False

    @staticmethod
    def reassign_features_pca(child_clusters, max_tries=None):
        if len(child_clusters) < 2:
            return child_clusters

        n_tries = 0

        # Loop through all features for all cluster combinations
        for i in range(len(child_clusters)):

            if len(child_clusters[i].features) == 1:
                continue

            for feature in child_clusters[i].features:
                for j in range(len(child_clusters)):

                    if i == j:
                        continue

                    remaining_clusters = list(set(child_clusters)
                                              - {child_clusters[i], child_clusters[j]})
                    print('there are {} remaining clusters'.format(len(remaining_clusters)))
                    child_clusters[i], child_clusters[j], change_flag = \
                        VarClus.reassign_one_feature_pca(child_clusters[i],
                                                         child_clusters[j],
                                                         feature,
                                                         remaining_clusters)
                    # TODO: log
                    if change_flag:
                        print('feature {} was re-assigned'.format(feature))
                        print('child_clusters[i] has {0} features and child_clusters[j] has {1} ' \
                              'features'.format(len(child_clusters[i].features),
                                                len(child_clusters[j].features)))

                    if not change_flag:
                        n_tries += 1

                    if max_tries and n_tries >= max_tries:
                        return child_clusters

        return child_clusters

    @staticmethod
    def nearest_component_sorting_once(initial_child_clusters):
        for cluster in initial_child_clusters:
            if not getattr(cluster, 'pca', False):
                cluster.run_pca()

        full_dataframe = pd.concat(
            [cluster.dataframe for cluster in initial_child_clusters],
            axis=1
        )
        corr_table = pd.concat(
            [full_dataframe.corrwith(cluster.pca_features[0]) for cluster in
             initial_child_clusters],
            axis=1
        )
        corr_sq_table = corr_table ** 2
        corr_max = corr_sq_table.max(axis=1)
        cluster_membership = corr_sq_table.apply(lambda x: x == corr_max)

        if (cluster_membership.sum() == 0).sum():
            print('Near orthogonal feature clusters detected in NCS phase. Randomizing...')
            i_range, j_range = cluster_membership.shape
            for i in range(i_range):
                for j in range(j_range):
                    cluster_membership.iloc[i, j] = (i % j_range == j)

        new_child_clusters = [
            Cluster(dataframe=full_dataframe,
                    n_split=initial_child_clusters[0].n_split,
                    features=[feature for (feature, condition)
                              in cluster_membership[membership].to_dict().items()
                              if condition],
                    parents=initial_child_clusters[0].parents,
                    name='{0}-{1}'.format(initial_child_clusters[0].parents[0].name, str(i)))
            for i, membership in enumerate(cluster_membership)
        ]

        # Check if clusters are unchanged
        old_cluster_features = set([
            tuple(cluster.features.sort() or cluster.features) for cluster in initial_child_clusters
        ])

        new_cluster_features = set([
            tuple(cluster.features.sort() or cluster.features) for cluster in new_child_clusters
        ])

        return new_child_clusters, old_cluster_features != new_cluster_features

    @staticmethod
    def nearest_component_sorting(initial_child_clusters, max_tries=None):
        n_tries = 0
        change_flag = True
        new_child_clusters = initial_child_clusters

        while change_flag:
            new_child_clusters, change_flag = \
                VarClus.nearest_component_sorting_once(new_child_clusters)

            n_tries += 1
            if max_tries and n_tries >= max_tries:
                break

        return new_child_clusters

    @staticmethod
    def one_step_decompose(cluster, max_tries=None):
        if not getattr(cluster, 'pca', False):
            cluster.run_pca()

        corr_table = pd.concat(cluster.pca_corr, axis=1)
        corr_sq_table = corr_table ** 2
        corr_max = corr_sq_table.max(axis=1)
        cluster_membership = corr_sq_table.apply(lambda x: x == corr_max)

        if (cluster_membership.sum() == 0).sum():
            print('Near orthogonal feature clusters detected in NCS phase. Randomizing...')
            i_range, j_range = cluster_membership.shape
            for i in range(i_range):
                for j in range(j_range):
                    cluster_membership.iloc[i, j] = (i % j_range == j)

        child_clusters = [
            Cluster(dataframe=cluster.dataframe,
                    n_split=cluster.n_split,
                    features=[feature for (feature, condition)
                              in cluster_membership[membership].to_dict().items()
                              if condition],
                    parents=[cluster],
                    name='{0}-{1}'.format(cluster.name, str(i)))
            for i, membership in enumerate(cluster_membership)
        ]

        # Phase 1: nearest component sorting
        print('phase #1: NCS')
        child_clusters = \
            VarClus.nearest_component_sorting(child_clusters, max_tries=max_tries)

        # Phase 2: search algorithm
        print('phase #2: Search')
        child_clusters = \
            VarClus.reassign_features_pca(child_clusters, max_tries=max_tries)

        return child_clusters

    @staticmethod
    def __decompose(cluster, max_eigenvalue, max_tries):
        if not getattr(cluster, 'pca', False):
            cluster.run_pca()

        if cluster.pca.explained_variance_[-1] >= max_eigenvalue and \
            len(cluster.features) >= cluster.n_split and \
            len(cluster.features) > 1:

            print('decomposing cluster with hash {}'.format(cluster.__hash__()))
            cluster.children = VarClus.one_step_decompose(cluster, max_tries=max_tries)

            for child_cluster in cluster.children:
                VarClus.__decompose(child_cluster,
                                    max_eigenvalue,
                                    max_tries)

    def decompose(self, dataframe):
        """
        Decomposes given dataframe in an oblique hierarchical way.

        :param dataframe: feature space that needs to be decomposed
        """

        self.cluster = Cluster(dataframe,
                               self.n_split,
                               name='cluster-0')

        VarClus.__decompose(self.cluster,
                            self.max_eigenvalue,
                            self.max_tries)

        return self.cluster

    @property
    def final_cluster_structure(self):
        if not getattr(self, 'cluster', False):
            print('cluster was not fitted yet')
            return dict()

        # TODO: make it a dict
        return self.cluster.return_all_leaves()




