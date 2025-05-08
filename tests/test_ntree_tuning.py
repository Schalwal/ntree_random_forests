# , tune_ntree_rf, tune_ntree_gb
from ntree_tuning import Ntree_RandForest_Classifier, Ntree_RandForest_Regressor, Ntree_GradBoost_Classifier, Ntree_GradBoost_Regressor
from sklearn.datasets import make_regression, make_classification
from . import RANDOM_STATE
import pytest
import numpy as np


def test_cls_tuner():
    Xcls, ycls = make_classification(
        n_samples=200, n_features=20, n_classes=3, random_state=RANDOM_STATE, n_clusters_per_class=3, n_informative=5)

    n_estimators = 100

    rf_cls = Ntree_RandForest_Classifier(n_estimators=n_estimators)
    rf_cls.fit(Xcls, ycls)
    gb_cls = Ntree_GradBoost_Classifier(
        n_estimators=n_estimators, subsample=0.8)
    gb_cls.fit(Xcls, ycls)

    min_trees = 20
    max_trees = 100
    delta_trees = 10

    rf_ntree_dict = rf_cls.tune_ntrees(
        Xcls, ycls, min_trees=min_trees, max_trees=max_trees, delta_trees=delta_trees)

    gb_ntree_dict = gb_cls.tune_ntrees()

    # check that dict is correct
    assert isinstance(rf_ntree_dict, dict)
    assert isinstance(gb_ntree_dict, dict)
    assert is_valid_dict(rf_ntree_dict)
    assert is_valid_dict(gb_ntree_dict)
    assert len(rf_ntree_dict) == 9
    assert all(0 <= value <= 1 for value in rf_ntree_dict.values())
    assert len(gb_ntree_dict) == n_estimators


def test_regression_tuner():
    Xreg, yreg = make_regression(
        n_samples=200, n_features=20, random_state=RANDOM_STATE)
    n_estimators = 100
    rf_reg = Ntree_RandForest_Regressor(n_estimators=n_estimators)
    rf_reg.fit(Xreg, yreg)
    gb_reg = Ntree_GradBoost_Regressor(
        n_estimators=n_estimators, subsample=0.8)
    gb_reg.fit(Xreg, yreg)

    min_trees = 20
    max_trees = 100
    delta_trees = 10

    rf_ntree_dict = rf_reg.tune_ntrees(Xreg, yreg, min_trees=min_trees,
                                       max_trees=max_trees, delta_trees=delta_trees)
    gb_ntree_dict = gb_reg.tune_ntrees()

    assert len(rf_ntree_dict) == 9
    assert len(gb_ntree_dict) == n_estimators


def test_is_model_fitted():

    Xcls, ycls = make_classification(
        n_samples=200, n_features=20, n_classes=3, random_state=RANDOM_STATE, n_clusters_per_class=3, n_informative=5)
    Xreg, yreg = make_regression(
        n_samples=200, n_features=20, random_state=RANDOM_STATE)

    rf_cls = Ntree_RandForest_Classifier()
    rf_reg = Ntree_RandForest_Regressor()
    gb_cls = Ntree_GradBoost_Classifier(subsample=0.8)
    gb_reg = Ntree_GradBoost_Regressor(subsample=0.8)

    error_message = 'The `estimators_` attribute is missing in the model. You probably have not fitted the model yet which however is necessary to call the method `tune_ntrees`'

    for rf_model in [rf_cls, rf_reg]:
        with pytest.raises(ValueError, match=error_message):
            rf_model.tune_ntrees(X=Xreg, y=yreg)

    for gb_model in [gb_cls, gb_reg]:
        with pytest.raises(ValueError, match=error_message):
            gb_model.tune_ntrees()


def is_valid_dict(d):
    """Helper function to check if a dictionary has integer keys and float values."""
    if not isinstance(d, dict):
        return False
    for key, value in d.items():
        if not isinstance(key, int) or not isinstance(value, float):
            return False
    return True
