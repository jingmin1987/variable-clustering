# variable-clustering
A re-creation of SAS Varclus procedure in Python. Varclus is a very useful decomposition and 
dimensionality reduction tool when a large dataframe with numerous features is dealt with. 

Varclus has several advantages and disadvantages over other decomposition and dimensionality 
reduction algorithms such as PCA and K-means
## Advantages
* Final results are real features, not made-up components
* Hierarchical structure that could be more meaningful to business
* Easy to use for feature swapping when needed (business requirement, regulatory constraint, etc.)

## Disadvantages
* Oblique decomposition which could yield highly correlated clusters
* Algorithmic expensive.
* Not optimal in dimensionality reduction

## Description of the algorithm  
The algorithm is a recurisve process that tries to decompose each child cluster based on the same
 critera (number of splits, max eigenvalue). Below is a detailed description of each recursion  
1. Conducts PCA on current feature space. If the max eigenvalue is smaller than the 
specified threshold, stop the decomposition process
2. Calculates the first N PCA components and assign features to these components based on
absolute correlation from high to low. These components are the initial centroids of
these child clusters.
3. After initial assignment, the algorithm conducts an iterative assignment called Nearest
Component Sorting (NCS). Basically, the centroid vectors are re-computed as the first
components of the child clusters and the algorithm will re-assign each of the feature
based on the same correlation rule.
4. After NCS, the algorithm tries to increase the total variance explained by the first PCA 
component of each child cluster by re-assigning features across clusters


## Other comments
* As the same with other unsupervised clustering methods, multiple hyperparameters need to be tuned
* Trade-off between art and science
* Right now, the algorithm only supports hierarchical decomposition and using first PCA component
 as the centroid. In SAS Proc Varclus, one can also specify global re-assignment and using mean 
 vector as the centroid
 * Please see demo.ipynb for more detailed information
 * Dependencies: pandas, np, sklearn