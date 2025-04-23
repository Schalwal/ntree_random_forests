from ntree_tuning import Ntree_RF_Classifier, Ntree_RF_Regressor, tune_ntree_rf
from sklearn.datasets import make_regression, make_classification
from . import RANDOM_STATE


def test_cls_tuner():
    Xcls, ycls = make_classification(
        n_samples=200, n_features=20, n_classes=3, random_state=RANDOM_STATE, n_clusters_per_class=3, n_informative=5)
    rf_cls = Ntree_RF_Classifier(n_estimators=100)
    rf_cls.fit(Xcls, ycls)

    min_trees = 20
    max_trees = 100
    delta_trees = 10

    tune_dict = tune_ntree_rf(
        rf_cls, Xcls, ycls, min_trees=min_trees, max_trees=max_trees, delta_trees=delta_trees)

    assert len(tune_dict) == 9
    assert all(0 <= value <= 1 for value in tune_dict.values())


def test_regression_tuner():
    Xreg, yreg = make_regression(
        n_samples=200, n_features=20, random_state=RANDOM_STATE)
    rf_reg = Ntree_RF_Regressor(n_estimators=100)
    rf_reg.fit(Xreg, yreg)

    min_trees = 20
    max_trees = 100
    delta_trees = 10

    tune_dict = tune_ntree_rf(rf_reg, Xreg, yreg, min_trees=min_trees,
                              max_trees=max_trees, delta_trees=delta_trees)

    assert len(tune_dict) == 9
