from . import Ntree_GB_Regressor, Ntree_GB_Classifier
from ._utils import validate_ntree_parameters
import numpy as np
from typing import Union


def tune_ntree_gb(gb_model: Union[Ntree_GB_Classifier, Ntree_GB_Regressor]):
    """Tune Gradient Boosting w.r.t. to n_trees considering the OOB error. For Regression it's the specified OOB loss and for Classification it's 1 - Accuracy."""

    # 1. check correct argument type
    if not isinstance(gb_model, Ntree_GB_Classifier) and not isinstance(gb_model, Ntree_GB_Regressor):
        raise TypeError(
            f"Expected an `rf_model` to be either `Ntree_GB_Classifier` or `Ntree_GB_Regressor`, but got {type(gb_model).__name__}.")

    if gb_model.subsample == 1:
        raise ValueError(
            "For tuning Gradient Boosting, you need to set `subsample` to <1"
        )

    # 2. ensure that `min_trees`, `delta_trees` and `max_trees` are correctly set
    # min_trees, max_trees, delta_trees = validate_ntree_parameters(
    #     gb_model, min_trees, max_trees, delta_trees)

    # 3. calculate oob_errors
    oob_errors = {i: oob_score.item() for (
        i, oob_score) in enumerate(gb_model.oob_scores_)}

    return oob_errors
