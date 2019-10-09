from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from joblib import Parallel, delayed
from time import time


class MLSVC:
    """
    Multi-Label Support Vector Machine, with individual optimizations per binary problem.
    """

    def __init__(self, n_jobs=1, estimator=LinearSVC, *args, **kwargs):
        self.n_jobs = n_jobs
        self.args = args
        self.kwargs = kwargs
        self.verbose = False if 'verbose' not in self.kwargs else self.kwargs['verbose']
        self.estimator = estimator


    def fit(self, X, y, **grid_search_params):
        tini = time()
        assert len(y.shape)==2 and set(np.unique(y).tolist()) == {0,1}, 'data format is not multi-label'
        nD,nC = y.shape
        prevalence = np.sum(y, axis=0)
        self.svms = np.array([self.estimator(*self.args, **self.kwargs) for _ in range(nC)])
        if grid_search_params and grid_search_params['param_grid']:
            self._print('grid_search activated with: {}'.format(grid_search_params))
            # Grid search cannot be performed if the category prevalence is less than the parameter cv.
            # In those cases we place a svm instead of a gridsearchcv
            cv = 5 if 'cv' not in grid_search_params else grid_search_params['cv']
            assert isinstance(cv, int), 'cv must be an int (other policies are not supported yet)'
            self.svms = [GridSearchCV(svm_i, refit=True, **grid_search_params) if prevalence[i]>=cv else svm_i
                         for i,svm_i in enumerate(self.svms)]
        for i in np.argwhere(prevalence==0).flatten():
            self.svms[i] = TrivialRejector()

        self.svms = Parallel(n_jobs=self.n_jobs)(
            delayed(self.svms[c].fit)(X,y[:,c]) for c,svm in enumerate(self.svms)
        )
        self.training_time = time() - tini


    def predict(self, X):
        return np.vstack(list(map(lambda svmi: svmi.predict(X), self.svms))).T


    def predict_proba(self, X):
        return np.vstack(map(lambda svmi: svmi.predict_proba(X)[:,np.argwhere(svmi.classes_==1)[0,0]], self.svms)).T


    def _print(self, msg):
        if self.verbose>0:
            print(msg)


    def best_params(self):
        return [svmi.best_params_ if isinstance(svmi, GridSearchCV) else None for svmi in self.svms]


class TrivialRejector:
    def fit(self,*args,**kwargs): return self
    def predict(self, X): return np.zeros(X.shape[0])
    def predict_proba(self, X): return np.zeros(X.shape[0])

