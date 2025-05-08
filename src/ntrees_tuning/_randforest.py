from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import is_classifier
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from scipy.sparse import issparse
import numpy as np
from ._utils import validate_ntree_parameters
from sklearn.ensemble._forest import BaseForest, _generate_unsampled_indices, _get_n_samples_bootstrap
from warnings import warn


# 1.1 RANDOM FOREST CLASSIFIER
class Ntree_RandForest_Classifier(RandomForestClassifier):
    """
    Subclass of sklearn.ensemble.RandomForestClassifier

    Please refer to the sklearn-documentation for the standard parameters here: 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    The only difference are the two extra methods `predict_ntrees` and `tune_ntree`.

    Methods (additional to sklearn):
    -------------------------------

        `predict_ntrees(X, ntrees, sample_random)`:
            This method is like the `predict` method only that it let's you choose how many trees of the forest should be used for prediction. The `sample_random` parameter lets you decide which how the trees should be selected. Read the method description for more details.

        `tune_ntree(X, y, min_trees, max_trees, delta_trees, sample_random, random_state)`:
            After fitting the model, this method will tune the `ntrees` parameter w.r.t the oob_error. It returns a dictionary with the `ntrees` values as keys and the oob-error as value. Read the method description for more details.
    """

    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)

    def predict_ntrees(self, X, ntrees=None, sample_random=True):
        """Generate predictions with the `Ntree_RandForest_Classifier` model on a dataset X and choose the number of trees used for that prediction.

        Args:
            X (np.ndarray): the data to predict on
            n_trees (int, optional): The number of trees used in the prediction. Defaults to None.
            sample_random (bool, optional): Choose the way the `ntrees` trees used for prediction should be chosen. False means that the first `ntrees` trees will be used. True means that `ntrees` trees will be randomly drawn. Defaults to False.
            random_state (int, optional): Choose a random state for the random sampling of trees. Only has an effect if `sample_random` is set to True. Defauls to None.

        Returns:
            np.ndarray: 1D-np.ndarray with predictions. As many predictions as observations in X.
        """
        if ntrees is None:
            ntrees = self.n_estimators  # Use all trees by default
        else:
            # Ensure we don't exceed the number of trees
            ntrees = min(ntrees, self.n_estimators)

        if sample_random:
            indices = np.random.choice(
                len(self.estimators_), size=ntrees, replace=False)
        else:
            indices = np.arange(ntrees)
        predictions = [tree.predict(X)
                       for tree in np.array(self.estimators_)[indices]]
        return self._majority_vote(predictions)

    def _majority_vote(self, predictions):
        predictions = np.array(predictions).astype(int)
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])

    def tune_ntrees(self, X: np.ndarray, y: np.ndarray, min_trees: int = 10, max_trees: int = None, delta_trees: int = 10,  sample_random: bool = False, random_state: int = None):
        """Tune the `Ntree_RandForest_Classifier` model w.r.t. n_trees considering the OOB error. The method returns a dictionary which maps each value of `ntrees` to the corresponding oob-errors. The oob-error is calculated as 1 - 'accuracy on the oob-samples'.

        Args:
            X (np.ndarray): the dataset
            y (np.ndarray): the target variables
            min_trees (int, optional): the smallest value for `ntrees` that should be used in tuning. Defaults to 10.
            max_trees (int, optional): the biggest value for `ntrees` that should be used in tuning. Defaults to None.
            delta_trees (int, optional): the step size for the ntree_value between `min_trees`and `max_trees`. Defaults to 10.
            sample_random (bool, optional): Choose the way the `ntrees` trees used for prediction should be chosen. False means that the first `ntrees` trees will be used. True means that `ntrees` trees will be randomly drawn. Defaults to False.
            random_state (int, optional): Choose a random state for the random sampling of trees. Only has an effect if `sample_random` is set to True. Defauls to None.

        Returns:
            dict[int, float]: dictionary which maps each value of `ntrees` (keys) to the corresponding oob-errors (values)
        """

        # ensure that `min_trees`, `delta_trees` and `max_trees` are correctly set
        min_trees, max_trees, delta_trees = validate_ntree_parameters(
            self, min_trees, max_trees, delta_trees)

        # check if model has been fitted already
        if not hasattr(self, 'estimators_'):
            raise ValueError(
                'The `estimators_` attribute is missing in the model. You probably have not fitted the model yet which however is necessary to call the method `tune_ntrees`')

        # calculate oob_errors (for classification)
        oob_errors = {}
        for n_trees in range(min_trees, max_trees+1, delta_trees):
            oob_preds = custom_compute_oob_predictions(
                rf=self,
                X=X,
                y=y,
                n_trees=n_trees,
                sample_random=sample_random,
                random_state=random_state
            ).squeeze()
            oob_preds = oob_preds.argmax(axis=1)
            oob_error = 1 - accuracy_score(oob_preds, y)
            oob_errors[n_trees] = oob_error
        return oob_errors

# 1.2 RANDOM FOREST REGRESSOR


class Ntree_RandForest_Regressor(RandomForestRegressor):
    """
    Subclass of sklearn.ensemble.RandomForestRegressor

    Please refer to the sklearn-documentation for the standard parameters here: 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    The only difference are the two extra methods `predict_ntrees` and `tune_ntree`.

    Methods (additional to sklearn):
    -------------------------------

        `predict_ntrees(X, ntrees, sample_random)`:
            This method is like the `predict` method only that it let's you choose how many trees of the forest should be used for prediction. The `sample_random` parameter lets you decide which how the trees should be selected. Read the method description for more details.

        `tune_ntree(X, y, min_trees, max_trees, delta_trees, sample_random, random_state)`:
            After fitting the model, this method will tune the `ntrees` parameter w.r.t the oob_error. It returns a dictionary with the `ntrees` values as keys and the oob-error as value. Read the method description for more details.
    """

    def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples)

    def predict_ntrees(self, X: np.ndarray, ntrees: int = None, sample_random: bool = False, random_state: int = None) -> np.ndarray:
        """Generate predictions with the `Ntree_RandForest_Regressor` model on a dataset X and choose the number of trees used for that prediction.

        Args:
            X (np.ndarray): the data to predict on
            n_trees (int, optional): The number of trees used in the prediction. Defaults to None.
            sample_random (bool, optional): Choose the way the `ntrees` trees used for prediction should be chosen. False means that the first `ntrees` trees will be used. True means that `ntrees` trees will be randomly drawn. Defaults to False.
            random_state (int, optional): Choose a random state for the random sampling of trees. Only has an effect if `sample_random` is set to True. Defauls to None.

        Returns:
            np.ndarray: 1D-np.ndarray with predictions. As many predictions as observations in X.
        """
        if ntrees is None:
            ntrees = self.n_estimators  # Use all trees by default
        else:
            # Ensure we don't exceed the number of trees
            ntrees = min(ntrees, self.n_estimators)

        if sample_random:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(
                len(self.estimators_), size=ntrees, replace=False)
        else:
            indices = np.arange(ntrees)

        N = X.shape[0]
        predictions = np.zeros((N, ntrees))
        for i, tree in enumerate(np.array(self.estimators_)[indices]):
            predictions[:, i] = tree.predict(X)

        return predictions.mean(axis=1)

    def tune_ntrees(self, X: np.ndarray, y: np.ndarray, min_trees: int = 10, max_trees: int = None, delta_trees: int = 10,  sample_random: bool = False, random_state: int = None) -> dict[int, float]:
        """Tune the `Ntree_RandForest_Regressor` model w.r.t. n_trees considering the OOB error. The method returns a dictionary which maps each value of `ntrees` to the corresponding oob-errors. The oob-error is calculated with the `criterion` of the `Ntree_RandForest_Regressor` model.

        Args:
            X (np.ndarray): the dataset
            y (np.ndarray): the target variables
            min_trees (int, optional): the smallest value for `ntrees` that should be used in tuning. Defaults to 10.
            max_trees (int, optional): the biggest value for `ntrees` that should be used in tuning. Defaults to None.
            delta_trees (int, optional): the step size for the ntree_value between `min_trees`and `max_trees`. Defaults to 10.
            sample_random (bool, optional): Choose the way the `ntrees` trees used for prediction should be chosen. False means that the first `ntrees` trees will be used. True means that `ntrees` trees will be randomly drawn. Defaults to False.
            random_state (int, optional): Choose a random state for the random sampling of trees. Only has an effect if `sample_random` is set to True. Defauls to None.

        Returns:
            dict[int, float]: dictionary which maps each value of `ntrees` (keys) to the corresponding oob-errors (values)
        """

        # ensure that `min_trees`, `delta_trees` and `max_trees` are correctly set
        min_trees, max_trees, delta_trees = validate_ntree_parameters(
            self, min_trees, max_trees, delta_trees)

        # check if model has been fitted already
        if not hasattr(self, 'estimators_'):
            raise ValueError(
                'The `estimators_` attribute is missing in the model. You probably have not fitted the model yet which however is necessary to call the method `tune_ntrees`')

        # calculate oob_errors (for regression)
        oob_errors = {}
        for n_trees in range(min_trees, max_trees+1, delta_trees):
            oob_preds = custom_compute_oob_predictions(
                rf=self,
                X=X,
                y=y,
                n_trees=n_trees,
                sample_random=sample_random,
                random_state=random_state
            ).squeeze()
            if self.criterion == 'squared_error':
                oob_error = mean_squared_error(oob_preds, y)
            elif self.criterion == 'absolute_error':
                oob_error = mean_absolute_error(oob_preds, y)
            else:
                warn('This function only supports oob_error-calculation with mean squared error or mean absolute error. OOB error now computed with MSE.')
                oob_error = mean_squared_error(oob_preds, y)

            oob_errors[n_trees] = oob_error
        return oob_errors


def custom_compute_oob_predictions(rf: BaseForest, X, y, n_trees, sample_random=False, random_state=None):
    """Compute oob predictions for given X and y taking n_trees into account. Slightly changed from sklearn."""

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

    # choose tree_indices depending on sample_random
    if sample_random:
        rng = np.random.default_rng(random_state)
        tree_indices = rng.choice(rf.n_estimators, size=n_trees, replace=False)
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
