from sklearn.ensemble._forest import BaseForest, _generate_unsampled_indices, _get_n_samples_bootstrap
from sklearn.base import is_classifier
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.sparse import issparse
from src.ntree_random_forest import Ntree_RF_Classifier, Ntree_RF_Regressor
from warnings import warn
import numpy as np
from collections import OrderedDict
from typing import Union



def tune_ntree_rf(rf_model: Union[Ntree_RF_Classifier, Ntree_RF_Regressor], max_trees: int, X: np.ndarray, y: np.ndarray, sample_random = False):
    """Tune Random Forest for CLASSIFICATION w.r.t. to n_trees"""
    
    # check correct argument type
    if not isinstance(rf_model, Ntree_RF_Regressor) and not isinstance(rf_model, Ntree_RF_Classifier):
        raise TypeError(f"Expected an `rf_model` to be either `Ntree_RF_Classifier` or `Ntree_RF_Regressor`, but got {type(rf_model).__name__}.")
    
    # calculate oob_errors
    oob_errors = {}
    for n_trees in range(1, max_trees+1):
        oob_preds = custom_compute_oob_predictions(rf_model, X, y, n_trees, sample_random=sample_random).squeeze()
        if isinstance(rf_model, Ntree_RF_Classifier):
            oob_preds = oob_preds.argmax(axis=1) # transformation for classification
            oob_error = 1 - accuracy_score(oob_preds, y)
        else:
            oob_error = mean_squared_error(oob_preds, y)
        oob_errors[n_trees] = oob_error
    return oob_errors 
      
      

def custom_compute_oob_predictions(rf: BaseForest, X, y, n_trees, sample_random = False): 
    """Compute oob predictions for given X and y"""
    
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
            n_classes = rf.n_classes_[0] # here rf.n_classes_ is a list
        
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
    
    # loop over trees
    if n_trees > rf.n_estimators:
        n_trees = rf.n_estimators
        warn(
                (
                    f"You set `n_trees (={n_trees})` bigger than n_estimators (={rf.n_estimators}). \n"
                    f"`n_trees` was now automatically set to n_estimators (={rf.n_estimators}) so that all trees are being used. "
                ),
                UserWarning,
            )
    
    if sample_random:
        tree_indices = np.random.choice(rf.n_estimators, size=n_trees, replace=False)
    else:
        tree_indices = np.arange(n_trees)
    
    for estimator in np.array(rf.estimators_)[tree_indices]: # only up to ntrees ... 
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