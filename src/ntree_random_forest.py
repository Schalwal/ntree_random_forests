from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


# 1.1 RANDOM FOREST CLASSIFIER 
class Ntree_RF_Classifier(RandomForestClassifier):
    
    def __init__(self, n_estimators=100, criterion='gini',max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)  

    def predict(self, X, n_trees=None, sample_random=True):
        if n_trees is None:
            n_trees = self.n_estimators  # Use all trees by default
        else:
            n_trees = min(n_trees, self.n_estimators)  # Ensure we don't exceed the number of trees
            
        if sample_random:
            indices = np.random.choice(len(self.estimators_), size=n_trees, replace=False)
        else:
            indices = np.arange(n_trees)
        predictions = [tree.predict(X) for tree in np.array(self.estimators_)[indices]]
        return self._majority_vote(predictions)

    def _majority_vote(self, predictions):
        predictions = np.array(predictions).astype(int)
        return np.array([np.bincount(pred).argmax() for pred in predictions.T])
    
    
# 1.2 RANDOM FOREST REGRESSOR 
class Ntree_RF_Regressor(RandomForestRegressor):
    
    def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease,bootstrap=bootstrap,oob_score=oob_score,n_jobs=n_jobs,random_state=random_state,verbose=verbose,warm_start=warm_start,ccp_alpha=ccp_alpha,max_samples=max_samples)  

    def predict(self, X, n_trees=None, sample_random=False):       
        if n_trees is None:
            n_trees = self.n_estimators  # Use all trees by default
        else:
            n_trees = min(n_trees, self.n_estimators)  # Ensure we don't exceed the number of trees
            
        if sample_random:
            indices = np.random.choice(len(self.estimators_), size=n_trees, replace=False)
        else:
            indices = np.arange(n_trees)
            
        N = X.shape[0]
        predictions = np.zeros((N, n_trees))
        for i, tree in enumerate(np.array(self.estimators_)[indices]):
            predictions[:, i] = tree.predict(X)
        
        return predictions.mean(axis=1)