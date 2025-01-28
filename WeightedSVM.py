from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import r2_score


class WeightedSVM:
    def __init__(self, kernel="rbf", C=1, gamma="scale", epsilon=0.1, alpha=1):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

    def fit(self, X, y, sample_weight):
        sample_weight = np.array(sample_weight) ** self.alpha
        self.estimator = SVR(
            kernel=self.kernel, C=self.C, gamma=self.gamma, epsilon=self.epsilon
        )
        self.estimator.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y_test):
        y_pred = self.estimator.predict(X)
        return r2_score(y_test, y_pred)

    def get_params(self, deep=True):
        return {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "weight_power": self.alpha,
        }
