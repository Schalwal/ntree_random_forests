from . import Ntree_GB_Regressor, Ntree_GB_Classifier
from .utils import validate_ntree_parameters
import numpy as np
from typing import Union


def tune_ntree_gb(gb_model: Union[Ntree_GB_Classifier, Ntree_GB_Regressor], X: np.ndarray, y: np.ndarray, min_trees: int = 10, max_trees: int = None, delta_trees: int = 10):
    """Tune Gradient BOosting for w.r.t. to n_trees considering the OOB error. For Regression it's the OOB-MSE and for Classification it's 1 - Accuracy."""

    # 1. check correct argument type
    if not isinstance(gb_model, Ntree_GB_Classifier) and not isinstance(gb_model, Ntree_GB_Regressor):
        raise TypeError(
            f"Expected an `rf_model` to be either `Ntree_GB_Classifier` or `Ntree_GB_Regressor`, but got {type(gb_model).__name__}.")

    if gb_model.subsample == 1:
        raise ValueError(
            "For tuning Gradient Boosting, you need to set subsample to <1"
        )

    # 2. ensure that `min_trees`, `delta_trees` and `max_trees` are correctly set
    min_trees, max_trees, delta_trees = validate_ntree_parameters(
        gb_model, min_trees, max_trees, delta_trees)

    # 3. calculate oob_errors
    oob_errors = gb_model.oob_scores_
    return oob_errors
