"""Official bounded classical algorithms for Tributo."""

from tributo_algorithms_classical.descriptors import (
    EXTRA_TREES_JOBLIB_DESCRIPTOR,
    EXTRA_TREES_NATIVE_DESCRIPTOR,
    LINEAR_REGRESSION_DESCRIPTOR,
    LOGISTIC_REGRESSION_DESCRIPTOR,
    RANDOM_FOREST_JOBLIB_DESCRIPTOR,
    RANDOM_FOREST_NATIVE_DESCRIPTOR,
)
from tributo_algorithms_classical.isolation_forest_descriptor import (
    ISOLATION_FOREST_DESCRIPTOR,
)
from tributo_algorithms_classical.kmeans_descriptor import (
    KMEANS_DESCRIPTOR,
    MINIBATCH_KMEANS_DESCRIPTOR,
)
from tributo_algorithms_classical.multinomial_nb_descriptor import (
    MULTINOMIAL_NB_DESCRIPTOR,
)
from tributo_algorithms_classical.pca_descriptor import PCA_DESCRIPTOR
from tributo_algorithms_classical.sgd_descriptor import (
    SGD_CLASSIFIER_DESCRIPTOR,
    SGD_REGRESSOR_DESCRIPTOR,
)

__all__ = [
    "EXTRA_TREES_JOBLIB_DESCRIPTOR",
    "EXTRA_TREES_NATIVE_DESCRIPTOR",
    "ISOLATION_FOREST_DESCRIPTOR",
    "KMEANS_DESCRIPTOR",
    "LINEAR_REGRESSION_DESCRIPTOR",
    "LOGISTIC_REGRESSION_DESCRIPTOR",
    "MINIBATCH_KMEANS_DESCRIPTOR",
    "MULTINOMIAL_NB_DESCRIPTOR",
    "PCA_DESCRIPTOR",
    "RANDOM_FOREST_JOBLIB_DESCRIPTOR",
    "RANDOM_FOREST_NATIVE_DESCRIPTOR",
    "SGD_CLASSIFIER_DESCRIPTOR",
    "SGD_REGRESSOR_DESCRIPTOR",
]
