from ntree_tuning import Ntree_RF_Classifier, Ntree_RF_Regressor, Ntree_GB_Classifier, Ntree_GB_Regressor, tune_ntree_rf, tune_ntree_gb
from sklearn.datasets import make_regression, make_classification
from . import RANDOM_STATE


def test_cls_tuner():
    Xcls, ycls = make_classification(
        n_samples=200, n_features=20, n_classes=3, random_state=RANDOM_STATE, n_clusters_per_class=3, n_informative=5)

    n_estimators = 100

    rf_cls = Ntree_RF_Classifier(n_estimators=n_estimators)
    rf_cls.fit(Xcls, ycls)
    gb_cls = Ntree_GB_Classifier(n_estimators=n_estimators, subsample=0.8)
    gb_cls.fit(Xcls, ycls)

    min_trees = 20
    max_trees = 100
    delta_trees = 10

    rf_ntree_dict = tune_ntree_rf(
        rf_cls, Xcls, ycls, min_trees=min_trees, max_trees=max_trees, delta_trees=delta_trees)

    gb_ntree_dict = tune_ntree_gb(gb_cls)

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
    rf_reg = Ntree_RF_Regressor(n_estimators=n_estimators)
    rf_reg.fit(Xreg, yreg)
    gb_reg = Ntree_GB_Regressor(n_estimators=n_estimators, subsample=0.8)
    gb_reg.fit(Xreg, yreg)

    min_trees = 20
    max_trees = 100
    delta_trees = 10

    rf_ntree_dict = tune_ntree_rf(rf_reg, Xreg, yreg, min_trees=min_trees,
                                  max_trees=max_trees, delta_trees=delta_trees)
    gb_ntree_dict = tune_ntree_gb(gb_reg)

    assert len(rf_ntree_dict) == 9
    assert len(gb_ntree_dict) == n_estimators


def is_valid_dict(d):
    """Helper function to check if a dictionary has integer keys and float values."""
    if not isinstance(d, dict):
        return False
    for key, value in d.items():
        if not isinstance(key, int) or not isinstance(value, float):
            return False
    return True
