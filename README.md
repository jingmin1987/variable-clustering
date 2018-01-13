# variable-clustering
A re-creation of SAS varclus procedure in Python. Varclus is a very useful decomposition and dimensionality reduction tool when a large dataframe with numerous features is dealt with. 

Varclus has several advantages and disadvantages over other decomposition and dimensionality reduction algorithms such as PCA and K-means
## Advantages
* Final results are real features, not made-up components
* Hierarchical structure that could be more meaningful to business
* Easy to use for feature swapping when needed (business requirement, regulatory constraint, etc.)

## Disadvantages
* Oblique decomposition which could yield highly correlated clusters
* Algorithmic expensive.
* Not optimal in dimensionality reduction

## Other comments
* As the same with other unsupervised clustering methods, multiple hyperparameters need to be tuned
* Trade-off between art and science

## Description of the algorithm
* ### Step 1
	Initial split
* ### Step 2
	Recursively re-assign features to maximize total variance explained by cluster components
* ### Step 3
	Iteratively split child clusters until stopping criteria is reached