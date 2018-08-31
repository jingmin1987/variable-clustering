"""
Class definitions used for VarClus
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from decomposition.base_class import BaseDecompositionClass


class Cluster:
    """
    A tree-node type container that is capable of decomposing itself based on PCA and holds the
    following information
        - features in this cluster
        - first n PCA components and their corresponding eigenvalues
    """

    def __init__(self, dataframe, n_split=2, feature_list=None, parents=None, children=None,
                 name=None):
        """

        :param dataframe: A pandas dataframe
        :param n_split: Number of sub-clusters every time a cluster is split
        :param feature_list: A list of feature names
        :param parents: A list of parents to this cluster, if any
        :param children: A list of children to this cluster, if any
        :param name: Name of the cluster
        """

        # Using dataframe.columns will generate an index-list which is not convertible to set
        self.features = feature_list or list(dataframe)
        self.dataframe = dataframe[self.features]
        self.n_split = n_split
        self.parents = parents or []
        self.children = children or []
        self.name = name or ''
        self.pca = None
        self.pca_features = []
        self.pca_corr = []

        self.input_check()

    def run_pca(self):
        """
        A wrapper around sklearn.decomposition.PCA.fit().

        Additionally, it calculates the first n_split PCA components

        :return:
        """

        if len(self.features) > self.n_split:
            print('Number of features is smaller than n_split, cannot conduct PCA')
            return

        self.pca = PCA(n_components=self.n_split)
        self.pca = self.pca.fit(self.dataframe)

        for i in range(self.n_split):
            self.pca_features.append(self.dataframe.dot(self.pca.components_[i]))
            self.pca_corr.append(self.dataframe.corrwith(self.pca_features[i]))

    def input_check(self):
        """
        Checks the input against below rules
            1. If the features is a list

        :return:
        """

        if type(self.features) is not list:
            print('Input argument features is not a list. Wrapping it in a list')
            self.features = [self.features]

    def return_all_leaves(self):
        """
        Returns all terminal child leaves. If no children, returns self

        :return: A list of terminal child leaves if any. Otherwise, returns [self]
        """

        if not self.children:
            return [self]

        child_leaves_nested = [child.return_all_leaves() for child in self.children]
        return [leaf for leaves in child_leaves_nested for leaf in leaves]

    def __key(self):
        return (tuple(self.features), self.dataframe.shape)

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())


class VarClus(BaseDecompositionClass):
    """
    A class that does oblique hierarchical decomposition of a feature space based on PCA.
    The general algorithm is
        1. Conducts PCA on current feature space. If the max eigenvalue is smaller than threshold,
            stop decomposition
        2. Calculates the first N PCA components and assign features to these components based on
            absolute correlation from high to low. These components are the initial centroids of
            these child clusters.
        3. After initial assignment, the algorithm conducts an iterative assignment called Nearest
            Component Sorting (NCS). Basically, the centroid vectors are re-computed as the first
            components of the child clusters and the algorithm will re-assign each of the feature
            based on the same correlation rule.
        4. After NCS, the algorithm tries to increase the total variance explained by the first
            PCA component of each child cluster by re-assigning features across clusters
    """

    def __init__(self, n_split=2, max_eigenvalue=1, max_tries=3):
        """

        :param n_split: Number of sub-clusters that every time a cluster is split into
        :param max_eigenvalue: Eigenvalue threshold below which the decomposition will be stopped
        :param max_tries: Number of max tries before the algorithm gives up
        """

        self.n_split = n_split
        self.max_eigenvalue = max_eigenvalue
        self.max_tries = max_tries
        self.cluster = None

    @staticmethod
    def reassign_one_feature_pca(cluster_from, cluster_to, feature, other_clusters=None):
        """
        Tries to re-assign a feature from a cluster to the other cluster to see if total
        explained variance of all clusters (represented by the first PCA component)is increased.
        If increased, the re-assignment will stay

        :param cluster_from: The cluster where the feature comes from
        :param cluster_to: The cluster where the feature will join
        :param feature: Feature to be tested
        :param other_clusters: Other clusters for calculating total explained variance
        :return: Original or new cluster_from and cluster_to
        """

        if not (feature in cluster_from.features):
            return cluster_from, cluster_to

        # This shouldn't happen when calling decompose()
        if feature in cluster_to.features:
            print('feature {} is already in cluster_to'.format(feature))
            return cluster_from, cluster_to

        print('assessing feature {}'.format(feature))

        for cluster in other_clusters:
            if cluster.pca is None:
                cluster.run_pca()

        other_clusters = other_clusters or []

        cluster_from_new_df = cluster_from.dataframe.drop(feature, axis=1)
        cluster_to_new_df = cluster_to.dataframe.join(cluster_from.dataframe[feature])
        cluster_from_new = Cluster(dataframe=cluster_from_new_df,
                                   n_split=cluster_from.n_split,
                                   parents=cluster_from.parents,
                                   name=cluster_from.name)
        cluster_to_new = Cluster(dataframe=cluster_to_new_df,
                                 n_split=cluster_to.n_split,
                                 parents=cluster_to.parents,
                                 name=cluster_to.name)

        # This shouldn't happen logically
        if len(cluster_from.features + cluster_to.features) != \
           len(cluster_from_new.features + cluster_to_new.features):
            missing_feature = set(cluster_from.features + cluster_to.features) - \
                set(cluster_from_new.features + cluster_to_new.features)
            print('feature missing....the missing feature is...{}').format(missing_feature)

        for cluster in [cluster_from, cluster_from_new, cluster_to, cluster_to_new]:
            if cluster.pca is None:
                cluster.run_pca()

        explained_variance_before_assignment = np.sum(
            [cluster.pca.explained_variance_[0] for cluster in
             [cluster_from, cluster_to] + other_clusters],
        )

        explained_variance_after_assignment = np.sum(
            [cluster.pca.explained_variance_[0] for cluster in
             [cluster_from_new, cluster_to_new] + other_clusters],
        )

        print('current EV is {0}, new EV is {1}'.format(explained_variance_before_assignment,
                                                        explained_variance_after_assignment))

        if explained_variance_after_assignment > explained_variance_before_assignment:
            return cluster_from_new, cluster_to_new, True
        else:
            return cluster_from, cluster_to, False

    @staticmethod
    def reassign_features_pca(child_clusters, max_tries=3):
        """
        Iteratively assesses if a re-assignment of a feature is going to increase the total
        variance explained of the child clusters. The variance explained by a child cluster is
        the variance explained by the first PCA component

        :param child_clusters: A list of clusters
        :param max_tries: Number of max tries before the algorithm gives up
        :return: New or original list of clusters
        """

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

                    other_clusters = \
                        list(set(child_clusters) - {child_clusters[i], child_clusters[j]})

                    child_clusters[i], child_clusters[j], change_flag = \
                        VarClus.reassign_one_feature_pca(child_clusters[i],
                                                         child_clusters[j],
                                                         feature,
                                                         other_clusters)
                    if change_flag:
                        print('Feature {} was re-assigned'.format(feature))
                        print('child_clusters[i] has {0} features and child_clusters[j] has {1} ' \
                              'features'.format(len(child_clusters[i].features),
                                                len(child_clusters[j].features)))

                    if not change_flag:
                        n_tries += 1

                    if max_tries and n_tries >= max_tries:
                        print('Number of max tries has been reached. Returning current result...')
                        return child_clusters

        return child_clusters

    @staticmethod
    def nearest_component_sorting_once(initial_child_clusters):
        """
        Updates the centroids of the initial child clusters and re-assigns the features to the
        clusters with updated centroids based on absolute correlation

        :param initial_child_clusters: A list of initial child clusters
        :return: A new list of child clusters and boolean indicating if the clusters have been
            updated or not
        """

        for cluster in initial_child_clusters:
            if cluster.pca is None:
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
            print('Features of this cluster are most correlated with first PCA component. Consider '
                  'increasing max_eigenvalue. Randomly assigning features to child clusters...')

            i_range, j_range = cluster_membership.shape
            for i in range(i_range):
                for j in range(j_range):
                    cluster_membership.iloc[i, j] = (i % j_range == j)

        new_child_clusters = [
            Cluster(dataframe=full_dataframe,
                    n_split=initial_child_clusters[0].n_split,
                    feature_list=[feature for (feature, condition)
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
    def nearest_component_sorting(initial_child_clusters):
        """
        Iteratively assigns features to the child clusters based on re-computed centroids of each
        child cluster

        :param initial_child_clusters: A list of initial child clusters
        :return: Updated list of child clusters
        """

        n_tries = 0
        change_flag = True
        new_child_clusters = initial_child_clusters

        while change_flag:
            new_child_clusters, change_flag = \
                VarClus.nearest_component_sorting_once(new_child_clusters)

        return new_child_clusters

    @staticmethod
    def one_step_decompose(cluster, max_tries=3):
        """
        Algorithm that conducts one-time decomposition of the cluster.

        :param cluster: A cluster to be decomposed
        :param max_tries: Number of max tries during re-assigning phase before it gives up
        :return: A list of child clusters of this cluster after decomposition
        """

        if cluster.pca is None:
            cluster.run_pca()

        corr_table = pd.concat(cluster.pca_corr, axis=1)
        corr_sq_table = corr_table ** 2
        corr_max = corr_sq_table.max(axis=1)
        cluster_membership = corr_sq_table.apply(lambda x: x == corr_max)

        if (cluster_membership.sum() == 0).sum():
            print('Features of this cluster are most correlated with first PCA component. Consider '
                  'increasing max_eigenvalue. Randomly assigning features to child clusters...')

            i_range, j_range = cluster_membership.shape
            for i in range(i_range):
                for j in range(j_range):
                    cluster_membership.iloc[i, j] = (i % j_range == j)

        child_clusters = [
            Cluster(dataframe=cluster.dataframe,
                    n_split=cluster.n_split,
                    feature_list=[feature for (feature, condition)
                                  in cluster_membership[membership].to_dict().items()
                                  if condition],
                    parents=[cluster],
                    name='{0}-{1}'.format(cluster.name, str(i)))
            for i, membership in enumerate(cluster_membership)
        ]

        # Phase 1: nearest component sorting
        print('phase #1: NCS')
        child_clusters = \
            VarClus.nearest_component_sorting(child_clusters)

        # Phase 2: search algorithm
        print('phase #2: Search')
        child_clusters = \
            VarClus.reassign_features_pca(child_clusters, max_tries=max_tries)

        return child_clusters

    @staticmethod
    def _decompose(cluster, max_eigenvalue, max_tries):
        """
        Main recursive function to decompose a feature space based on certain rules.

        :param cluster: An instance of Cluster class that represents a feature space
        :param max_eigenvalue: Eigenvalue threshold below which the decomposition will be stopped
        :param max_tries: Max number of tries when re-assigning features before it gives up
        :return:
        """

        if cluster.pca is None:
            cluster.run_pca()

        if cluster.pca.explained_variance_[-1] >= max_eigenvalue and \
           len(cluster.features) >= cluster.n_split and \
           len(cluster.features) > 1:

            print('decomposing cluster {}'.format(cluster.name))
            cluster.children = VarClus.one_step_decompose(cluster, max_tries=max_tries)

            for child_cluster in cluster.children:
                VarClus._decompose(child_cluster,
                                   max_eigenvalue,
                                   max_tries)

    def decompose(self, dataframe):
        """
        Scales and decomposes a given dataframe in an oblique hierarchical way.

        :param dataframe: a pandas dataframe that contains the feature space
        """

        scaled_dataframe = pd.DataFrame(scale(dataframe), columns=dataframe.columns)

        self.cluster = Cluster(scaled_dataframe,
                               self.n_split,
                               name='cluster-0')

        VarClus._decompose(self.cluster,
                           self.max_eigenvalue,
                           self.max_tries)

        return self.cluster

    @property
    def final_cluster_structure(self):
        """
        Gets the final cluster structure after decomposition

        :return:
        """

        if not self.cluster:
            print('Please decompose the feature space first. Empty ')
            return []

        return self.cluster.return_all_leaves()

    @staticmethod
    def _print_cluster_structure(root_cluster, prefix='', h_space=5):
        """
        Prints the hierarchical structure below a given root cluster

        :param root_cluster: An instance of Cluster class
        :param prefix: String to be prefixed for each print
        :param h_space: Horizontal spacing or indentation for each sub level
        :return: A printout that shows the decomposition of the feature space
        """

        if root_cluster.children:
            for index, child_cluster in enumerate(root_cluster.children):
                print(prefix + '|')
                print(prefix + '|' + '-' * h_space + child_cluster.name)

                if index == len(root_cluster.children) - 1:
                    new_prefix = prefix + ' ' * (h_space + 1)
                else:
                    new_prefix = prefix + '|' + ' ' * h_space

                VarClus._print_cluster_structure(child_cluster, prefix=new_prefix, h_space=h_space)
        else:
            for feature in root_cluster.features:
                print(prefix + '|' + '-' * h_space + feature)

    def print_cluster_structure(self, h_space=5):
        """
        Prints the hierarchical structure of the decomposition

        :param h_space: Horizontal spacing or indentation for each sub level
        :return: A printout that shows the decomposition of the feature space
        """

        print(self.cluster.name)
        VarClus._print_cluster_structure(self.cluster, h_space=h_space)


