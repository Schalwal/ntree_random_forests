from sklearn.ensemble._forest import BaseForest, _generate_unsampled_indices, _get_n_samples_bootstrap
from sklearn.base import is_classifier
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.sparse import issparse
from src.ntree_tuning import Ntree_RF_Classifier, Ntree_RF_Regressor
from warnings import warn
import numpy as np
from collections import OrderedDict
from typing import Union


def tune_ntree_rf(rf_model: Union[Ntree_RF_Classifier, Ntree_RF_Regressor], X: np.ndarray, y: np.ndarray, min_trees: int = 10, max_trees: int = None, delta_trees: int = 10,  sample_random=False):
    """Tune Random Forest for w.r.t. to n_trees considering the OOB error. For Regression it's the OOB-MSE and for Classification it's 1 - Accuracy."""

    # 1. check correct argument type
    if not isinstance(rf_model, Ntree_RF_Regressor) and not isinstance(rf_model, Ntree_RF_Classifier):
        raise TypeError(
            f"Expected an `rf_model` to be either `Ntree_RF_Classifier` or `Ntree_RF_Regressor`, but got {type(rf_model).__name__}.")

    # 2. ensure that `min_trees`, `delta_trees` and `max_trees` are correctly set
    min_trees, max_trees, delta_trees = validate_tree_parameters(
        rf_model, min_trees, max_trees, delta_trees)

    # 3. calculate oob_errors
    oob_errors = {}
    for n_trees in range(min_trees, max_trees+1, delta_trees):
        oob_preds = custom_compute_oob_predictions(
            rf_model, X, y, n_trees, sample_random=sample_random).squeeze()
        if isinstance(rf_model, Ntree_RF_Classifier):
            # transformation for classification
            oob_preds = oob_preds.argmax(axis=1)
            oob_error = 1 - accuracy_score(oob_preds, y)
        else:
            oob_error = mean_squared_error(oob_preds, y)
        oob_errors[n_trees] = oob_error
    return oob_errors


def custom_compute_oob_predictions(rf: BaseForest, X, y, n_trees, sample_random=False):
    """Compute oob predictions for given X and y"""

    # Prediction requires X to be in CSR format
    X = X.astype(np.float32)
    if issparse(X):
        X = X.tocsr()

    # get shapes
    n_samples = y.shape[0]
    n_outputs = rf.n_outputs_
    if is_classifier(rf) and hasattr(rf, "n_classes_"):
        # n_classes_ is a ndarray at this stage
        # all the supported type of target will have the same number of
        # classes in all outputs
        if isinstance(rf.n_classes_, int):
            n_classes = rf.n_classes_
        else:
            n_classes = rf.n_classes_[0]  # here rf.n_classes_ is a list

        oob_pred_shape = (n_samples, n_classes, n_outputs)
    else:
        # for regression, n_classes_ does not exist and we create an empty
        # axis to be consistent with the classification case and make
        # the array operations compatible with the 2 settings
        oob_pred_shape = (n_samples, 1, n_outputs)

    oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
    n_oob_pred = np.zeros((n_samples, n_outputs), dtype=np.int64)

    n_samples_bootstrap = _get_n_samples_bootstrap(
        n_samples,
        rf.max_samples,
    )

    if n_trees > rf.n_estimators:
        n_trees = rf.n_estimators
        warn(
            (
                f"You set `n_trees (={n_trees})` bigger than n_estimators (={rf.n_estimators}). \n"
                f"`n_trees` was now automatically set to n_estimators (={rf.n_estimators}) so that all trees are being used. "
            ),
            UserWarning,
        )

    if sample_random:
        tree_indices = np.random.choice(
            rf.n_estimators, size=n_trees, replace=False)
    else:
        tree_indices = np.arange(n_trees)

    # loop only up to ntrees ...
    for estimator in np.array(rf.estimators_)[tree_indices]:
        unsampled_indices = _generate_unsampled_indices(
            estimator.random_state,
            n_samples,
            n_samples_bootstrap,
        )

        y_pred = rf._get_oob_predictions(estimator, X[unsampled_indices, :])
        oob_pred[unsampled_indices, ...] += y_pred
        n_oob_pred[unsampled_indices, :] += 1

    for k in range(n_outputs):
        if (n_oob_pred == 0).any():
            warn(
                (
                    "Some inputs do not have OOB scores. This probably means "
                    "too few trees were used to compute any reliable OOB "
                    "estimates."
                ),
                UserWarning,
            )
            n_oob_pred[n_oob_pred == 0] = 1
        oob_pred[..., k] /= n_oob_pred[..., [k]]

    return oob_pred


def validate_tree_parameters(rf_model: Union[Ntree_RF_Classifier, Ntree_RF_Regressor], min_trees, max_trees, delta_trees):
    """
    Validates the parameters for tuning the n_trees in a random forest.

    Parameters:
    - min_trees (int): Minimum number of trees.
    - max_trees (int): Maximum number of trees.
    - delta_trees (int): Step size for the number of trees.

    Raises:
    - ValueError: If any of the parameters are invalid.
    """

    # Set max_trees to n_estimators of model if not specified
    n_estimators = rf_model.n_estimators
    if max_trees is None:
        max_trees = n_estimators
    elif max_trees > n_estimators:
        max_trees = n_estimators
        warn(
            f"You set `max_trees` = {max_trees} too high. It was set to {n_estimators}, the total number of trees in the model.")

    # Check if min_trees is a non-negative integer
    if not isinstance(min_trees, int) or min_trees < 0:
        raise ValueError("min_trees must be a non-negative integer.")

    # Check if max_trees is a positive integer
    if not isinstance(max_trees, int) or max_trees <= 0:
        raise ValueError("max_trees must be a positive integer.")

    # Check if delta_trees is a positive integer
    if not isinstance(delta_trees, int) or delta_trees <= 0:
        raise ValueError("delta_trees must be a positive integer.")

    # Check if min_trees is less than or equal to max_trees
    if min_trees > max_trees:
        raise ValueError("min_trees must be less than or equal to max_trees.")

    return min_trees, max_trees, delta_trees
