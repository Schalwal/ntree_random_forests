from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
from warnings import warn

# 2.1 GB CLASSIFIER


class Ntree_GradBoost_Classifier(GradientBoostingClassifier):
    """
    Subclass of sklearn.ensemble.GradientBoostingClassifier

    Please refer to the sklearn-documentation for the standard parameters here: 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

    The only difference are the two extra methods `predict_ntrees` and `tune_ntree`.

    **Note that you MUST provide a value for `subsample` < 1 s.t. there can be oob-samples.**

    Methods (additional to sklearn):
    -------------------------------

        `predict_ntrees(X, ntrees)`:
            This method is like the `predict` method only that it let's you choose how many trees of the forest should be used for prediction. Read the method description for more details.

        `tune_ntree()`:
            After fitting the model, this method will tune the `ntrees` parameter w.r.t the oob_error. It returns a dictionary with the `ntrees` values as keys and the oob-error as value. Read the method description for more details.
    """

    def __init__(self, subsample, loss='log_loss', learning_rate=0.1, n_estimators=100, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha
                         )
        if subsample == 1:
            ValueError(
                "In order to tune `ntrees` w.r.t. to the oob-error you need to set `subsample` < 1.")

    def predict_ntrees(self, X, ntrees=None):
        """
        Generate predictions with the `Ntree_GradBoost_Classifier` model on a dataset X and choose the number of trees used for that prediction.

        Args:
            X (np.ndarray): the data to predict on
            n_trees (int, optional): The number of trees used in the prediction. Defaults to None.

        Returns:
            np.ndarray: 1D-np.ndarray with predictions. As many predictions as observations in X.
        """
        if ntrees is None:
            ntrees = self.n_estimators  # Use all trees by default
        else:
            # Ensure we don't exceed the number of trees
            ntrees = min(ntrees, self.n_estimators)

        trees = self.estimators_[:ntrees]
        N_trees, n_classes = trees.shape
        N = X.shape[0]
        output = np.zeros((N, n_classes))
        for i in range(N_trees):
            for j in range(n_classes):
                predictions = trees[i, j].predict(X)
                output[:, j] += predictions

        predictions = np.argmax(output, axis=1)
        return predictions

    def tune_ntrees(self):
        """
        Tune the `Ntree_RandForest_Classifier` model w.r.t. n_trees considering the OOB error. The method returns a dictionary which maps each value of `ntrees` to the corresponding oob-errors. The oob-error is calculated as 1 - 'accuracy on the oob-samples'.

        Returns:
            dict[int, float]: dictionary which maps each value of `ntrees` (keys) to the corresponding oob-errors (values)
        """

        if self.subsample == 1:
            raise ValueError(
                "For tuning Gradient Boosting, you need to set `subsample` to <1"
            )

        # check if model has been fitted already
        if not hasattr(self, 'estimators_'):
            raise ValueError(
                'The `estimators_` attribute is missing in the model. You probably have not fitted the model yet which however is necessary to call the method `tune_ntrees`')

        oob_errors = {i: oob_score.item() for (
            i, oob_score) in enumerate(self.oob_scores_)}

        return oob_errors
# 2.2 GB REGRESSOR


class Ntree_GradBoost_Regressor(GradientBoostingRegressor):
    """
    Subclass of sklearn.ensemble.GradientBoostingRegressor

    Please refer to the sklearn-documentation for the standard parameters here: 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

    The only difference are the two extra methods `predict_ntrees` and `tune_ntree`.

    **Note that you MUST provide a value for `subsample` < 1 s.t. there can be oob-samples.**

    Methods (additional to sklearn):
    -------------------------------

        `predict_ntrees(X, ntrees)`:
            This method is like the `predict` method only that it let's you choose how many trees of the forest should be used for prediction. Read the method description for more details.

        `tune_ntree()`:
            After fitting the model, this method will tune the `ntrees` parameter w.r.t the oob_error. It returns a dictionary with the `ntrees` values as keys and the oob-error as value. Read the method description for more details.
    """

    def __init__(self, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease,
                         init=init, random_state=random_state, max_features=max_features, alpha=alpha, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha
                         )

    def predict_ntrees(self, X, ntrees=None):
        """
        Generate predictions with the `Ntree_GradBoost_Regressor` model on a dataset X and choose the number of trees used for that prediction.

        Args:
            X (np.ndarray): the data to predict on
            ntrees (int, optional): The number of trees used in the prediction. Defaults to None.

        Returns:
            np.ndarray: 1D-np.ndarray with predictions. As many predictions as observations in X.
        """
        if ntrees is None:
            ntrees = self.n_estimators  # Use all trees by default
        else:
            # Ensure we don't exceed the number of trees
            ntrees = min(ntrees, self.n_estimators)
        trees = self.estimators_[:ntrees]
        predictions = np.array([tree[0].predict(X)
                               for tree in trees]).sum(axis=0)
        return predictions

    def tune_ntrees(self):
        """
        Tune the `Ntree_RandForest_Classifier` model w.r.t. n_trees considering the OOB error. The method returns a dictionary which maps each value of `ntrees` to the corresponding oob-errors. The oob-error is calculated with the `criterion` of the `Ntree_GradBoost_Regressor` model.

        Returns:
            dict[int, float]: dictionary which maps each value of `ntrees` (keys) to the corresponding oob-errors (values)
        """

        # check if model has been fitted already
        if not hasattr(self, 'estimators_'):
            raise ValueError(
                'The `estimators_` attribute is missing in the model. You probably have not fitted the model yet which however is necessary to call the method `tune_ntrees`')

        if self.subsample == 1:
            raise ValueError(
                "For tuning Gradient Boosting, you need to set `subsample` to <1"
            )

        oob_errors = {i: oob_score.item() for (
            i, oob_score) in enumerate(self.oob_scores_)}

        return oob_errors
