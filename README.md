## Example usage:


# 1. Create data:

```python
from sklearn.datasets import make_classification, make_regression
Xcls, ycls = make_classification(n_samples=200, n_features=20, n_classes=3, random_state=42, n_clusters_per_class=3, n_informative=5)
Xreg, yreg = make_regression(n_samples=200, n_features=20, random_state=42)
```

# 2. Create and Fit RandomForest and GradientBoosting models for Regression and Classification

For the Ntree Tuning new custom classes are introduced. They are direct descendants of `sklearn` classes (`RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`).

```python
import ntree_tuning as ntt

rf_cls = ntt.Ntree_RF_Classifier(n_estimators=100)
rf_cls.fit(Xcls, ycls)

rf_reg = ntt.Ntree_RF_Regressor(n_estimators=100)
rf_reg.fit(Xreg, yreg)

gb_cls = ntt.Ntree_GB_Classifier(n_estimators=100, subsample=0.8)
gb_cls.fit(Xcls, ycls)

gb_reg = ntt.Ntree_GB_Regressor(n_estimators=100, subsample=0.8)
gb_reg.fit(Xreg, yreg)

```

# 3. Tune ntree

Depending on whether you have a RF or GB model you can now call `tune_ntree_rf` or `tune_ntree_gb` to get the oob_error dicts for the specified values of `ntrees`

```python

# Gradient Boosting
print(ntt.tune_ntree_gb(gb_reg))
print(ntt.tune_ntree_gb(gb_cls))


# Random Forests

min_trees = 20
max_trees = 80
delta_trees = 5

print(ntt.tune_ntree_rf(rf_reg, Xreg, yreg, min_trees, max_trees, delta_trees))
print(ntt.tune_ntree_rf(rf_cls, Xcls, ycls, min_trees, max_trees, delta_trees))

```


