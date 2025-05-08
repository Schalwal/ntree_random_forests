The package **ntrees_tuning** is an **extension to sklearn**. To Random Forests and Gradient Boosting it adds the `ntrees` parameter which gives control over how many trees are used for prediction. The main benefit is that it enables to tune the `ntrees` parameter w.r.t. the OOB-error without having to retrain a new model for each value of `ntrees`.

The package introduces subclasses to the `sklearn`-classes of Random Forest and Gradient Boosting (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor). Each adds two new methods called `predict_ntree` and `tune_ntree` which enable predicting and tuning the `ntrees` parameter possible

## Example usage:


# 1. Create data:

```python
from sklearn.datasets import make_classification, make_regression
Xcls, ycls = make_classification(n_samples=200, n_features=20, n_classes=3, random_state=42, n_clusters_per_class=3, n_informative=5)
Xreg, yreg = make_regression(n_samples=200, n_features=20, random_state=42)
```

# 2. Create and Fit RandomForest and GradientBoosting models for Regression and Classification

For tuning the `ntrees` parameter new custom classes are introduced. They are direct descendants of `sklearn` classes (`RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`).

```python
import ntree_tuning as ntt

rf_cls = ntt.Ntree_RandForest_Classifier(n_estimators=100)
rf_cls.fit(Xcls, ycls)

rf_reg = ntt.Ntree_RandForest_Regressor(n_estimators=100)
rf_reg.fit(Xreg, yreg)

gb_cls = ntt.Ntree_GradBoost_Classifier(n_estimators=100, subsample=0.8)
gb_cls.fit(Xcls, ycls)

gb_reg = ntt.Ntree_GradBoost_Regressor(n_estimators=100, subsample=0.8)
gb_reg.fit(Xreg, yreg)
```

# 3. Tune ntrees

You then can call the `tune_ntrees` method to get a dictionary of the pairs of the `ntrees` value and the oob-error.

```python
# Gradient Boosting
print(gb_reg.tune_ntrees())
print(gb_cls.tune_ntrees())


# Random Forests
min_trees = 20
max_trees = 80
delta_trees = 5

print(rf_reg.tune_ntrees(Xreg, yreg, min_trees, max_trees, delta_trees))
print(rf_cls.tune_ntrees(Xcls, ycls, min_trees, max_trees, delta_trees))
```

# 4. Predict with ntrees

```python
print(gb_reg.predict_ntrees(Xreg, ntrees=10))
```


