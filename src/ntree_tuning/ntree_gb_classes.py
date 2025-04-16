from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np

# 2.1 GB CLASSIFIER
class Ntree_GB_Classifier(GradientBoostingClassifier):
    
    def __init__(self, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha
) 

    def predict(self, X, n_trees=None):
        if n_trees is None:
            n_trees = self.n_estimators  # Use all trees by default
        else:
            n_trees = min(n_trees, self.n_estimators)  # Ensure we don't exceed the number of trees

        trees = self.estimators_[:n_trees]
        N_trees, n_classes = trees.shape
        N = X.shape[0]
        output = np.zeros((N, n_classes))
        for i in range(N_trees):
            for j in range(n_classes):
                predictions = trees[i, j].predict(X)
                output[:, j] += predictions
        
        predictions = np.argmax(output, axis=1)
        return predictions
    
# 2.2 GB REGRESSOR 
class Ntree_GB_Regressor(GradientBoostingRegressor):
    
    def __init__(self, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(loss = loss, learning_rate = learning_rate, n_estimators = n_estimators, subsample = subsample, criterion = criterion, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, max_depth = max_depth, min_impurity_decrease = min_impurity_decrease,
init = init, random_state = random_state, max_features = max_features, alpha = alpha, verbose = verbose, max_leaf_nodes = max_leaf_nodes, warm_start = warm_start, validation_fraction = validation_fraction, n_iter_no_change = n_iter_no_change, tol = tol, ccp_alpha = ccp_alpha
) 

    def predict(self, X, n_trees=None):
        if n_trees is None:
            n_trees = self.n_estimators  # Use all trees by default
        else:
            n_trees = min(n_trees, self.n_estimators)  # Ensure we don't exceed the number of trees

        trees = self.estimators_[:n_trees]
        predictions = np.array([tree[0].predict(X) for tree in trees]).sum(axis=0)
        
        return predictions