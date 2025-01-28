# very basic slow random forest

from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import r2_score


class osRandomForest:
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        n_jobs=None,
        ccp_alpha=0.0,
        random_state=None,
        output_smear=1,
        max_samples=1,
        power_smear=1,
        input_smear=0,
    ):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = (ccp_alpha,)
        self.max_samples = max_samples
        self.output_smear = output_smear
        self.probs_power = power_smear
        self.input_smear = input_smear

        self.rg = np.random.default_rng(random_state)
        if random_state is None:
            self.random_state = self.rg.integers(0, 2**32 - 1)
        else:
            self.random_state = random_state

    def fit_tree(self, X, y, probs, i, sample_weight):
        indexes = self.rg.choice(
            range(len(X)), size=int(self.max_samples * len(X)), replace=self.bootstrap
        )
        y_mapped = self.rg.normal(y, self.output_smear * np.std(y) * probs)
        X_boot = X[indexes]
        y_boot = y_mapped[indexes]
        sample_weight_boot = sample_weight[indexes]

        base_tree = DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state + i,
        )

        return base_tree.fit(X_boot, y_boot, sample_weight=sample_weight_boot)

    def fit(self, X, y, probs=None, sample_weight=None):
        if probs is None or np.sum(probs) == 0:
            probs = np.ones(len(X))

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        probs = np.array(probs) ** self.probs_power
        probs = probs / np.mean(probs)

        X = np.array(X)
        y = np.array(y)

        assert len(X) == len(y) == len(probs)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self.fit_tree)(X, y, probs, i, sample_weight)
            for i in range(self.n_estimators)
        )

    def predict(self, X):
        X = np.array(X)
        predictions = np.array(
            [m.predict(X) for m in self.estimators_]
        )  # could parrallelize this
        return np.mean(predictions, axis=0)

    def get_weights(self, x):
        preds = [m.predict(x.reshape(1, -1)) for m in self.estimators_]
        p, counts = np.unique(preds, return_counts=True)
        return p, counts

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return r2_score(y_test, predictions)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "output_smear": self.output_smear,
            "power_smear": self.probs_power,
            "input_smear": self.input_smear,
        }
