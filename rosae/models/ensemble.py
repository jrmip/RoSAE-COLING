import numpy as np

from pyod.models.combination import average, maximization, median

from pyod.utils import standardizer
from pyod.models.base import BaseDetector

from joblib import Parallel, delayed

import warnings

class Ensemble(BaseDetector):
    """ Ensemble class is highly inspired of pyod code from Yue Zhao. We perform ensemble
    processing such as his guidelines.

    """
    def __init__(self,
                 base_detectors: list,
                 combination: str = "median",
                 contamination=0.1,
                 n_jobs: int = 1,
                 verbose: int = 0):
        super(Ensemble, self).__init__(contamination=contamination)

        self.combination = combination
        self.base_detectors = base_detectors
        self.n_estimators = len(self.base_detectors)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        warnings.filterwarnings('ignore')

        if self.n_jobs == 1:
            for base in self.base_detectors:
                base.fit(X=X, y=y)
        else:
            X_cpy = [np.copy(X) for _ in range(self.n_estimators)]

            self.base_detectors = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(m.fit)(X=X_cpy[i], y=y) for i, m in enumerate(self.base_detectors)
            )

        ensemble_scores = np.zeros([X.shape[0], self.n_estimators])
        for i in range(self.n_estimators):
            ensemble_scores[:, i] = self.base_detectors[i].decision_scores_

        # z-normalisation (mean of values is 0 and standard deviation is 1)
        #  the following line is from pyod
        self.decision_score_mat, self.score_scalar_ = standardizer(ensemble_scores, keep_scalar=True)

        if self.combination == 'average':
            decision_score = average(self.decision_score_mat)
        elif self.combination == "maximisation":
            decision_score = maximization(self.decision_score_mat)
        else:
            decision_score = median(self.decision_score_mat)

        assert (len(decision_score) == X.shape[0])

        self.decision_scores_ = decision_score.ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, X, n_detectors: int = -1, X_tr=None):
        warnings.filterwarnings('ignore')

        if n_detectors != self.n_estimators and n_detectors != -1:
            estimators = np.random.choice(self.base_detectors, n_detectors)

            ensemble_scores = np.zeros([X_tr.shape[0], n_detectors])

            for i, base in enumerate(estimators):
                ensemble_scores[:, i] = base.decision_function(X_tr)
            
            _, score_scalar_ = standardizer(ensemble_scores, keep_scalar=True)

            ensemble_scores = np.zeros([X.shape[0], n_detectors])

            for i, base in enumerate(estimators):
                ensemble_scores[:, i] = base.decision_function(X)
            
            pred = score_scalar_.transform(ensemble_scores)
        else:
            ensemble_scores = np.zeros([X.shape[0], self.n_estimators])

            for i, base in enumerate(self.base_detectors):
                ensemble_scores[:, i] = base.decision_function(X)

            pred = self.score_scalar_.transform(ensemble_scores)

        if self.combination == 'average':
            decision_score = average(pred)
        elif self.combination == "maximisation":
            decision_score = maximization(pred)
        else:
            decision_score = median(pred)

        assert (len(decision_score) == X.shape[0])

        return decision_score.ravel()
    