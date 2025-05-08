from sklearn.ensemble import BaseEnsemble
from warnings import warn


def validate_ntree_parameters(ntree_model: BaseEnsemble, min_trees, max_trees, delta_trees):
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
    n_estimators = ntree_model.n_estimators
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
