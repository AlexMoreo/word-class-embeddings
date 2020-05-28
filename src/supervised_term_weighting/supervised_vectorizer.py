import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from supervised_term_weighting.tsr_functions import *


"""
Supervised Term Weighting function based on any Term Selection Reduction (TSR) function (e.g., information gain,
chi-square, etc.) or, more generally, on any function that could be computed on the 4-cell contingency table for
each category-feature pair.
The supervised_4cell_matrix (a CxF matrix containing the 4-cell contingency tables
for each category-feature pair) can be pre-computed (e.g., during the feature selection phase) and passed as an
argument.
When C>1, i.e., in multiclass scenarios, a global_policy is used in order to determine a single feature-score which
informs about its relevance. Accepted policies include "max" (takes the max score across categories), "ave" and "wave"
(take the average, or weighted average, across all categories -- weights correspond to the class prevalence), and "sum"
(which sums all category scores).
"""
class TSRweighting(BaseEstimator,TransformerMixin):
    def __init__(self, tsr_function, global_policy='max', supervised_4cell_matrix=None, sublinear_tf=True, norm='l2', n_jobs=-1):
        if global_policy not in ['max', 'ave', 'wave', 'sum']:
            raise ValueError('Global policy should be in {"max", "ave", "wave", "sum"}')
        self.tsr_function = tsr_function
        self.global_policy = global_policy
        self.supervised_4cell_matrix = supervised_4cell_matrix
        self.n_jobs = n_jobs
        self.sublinear_tf = sublinear_tf
        self.norm = norm

    def fit(self, X, y):
        # X is a csr_matrix co-occurrence matrix (e.g., as obtained by CountVectorizer)
        # y is the document-by-label matrix of shape (ndocs, nclass)
        self.unsupervised_vectorizer = \
            TfidfTransformer(norm=None, use_idf=False, smooth_idf=False, sublinear_tf=self.sublinear_tf).fit(X)

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        nD, nC = y.shape
        nF = X.shape[1]

        if self.supervised_4cell_matrix is None:
            self.supervised_4cell_matrix = get_supervised_matrix(X, y, n_jobs=self.n_jobs)
        else:
            if self.supervised_4cell_matrix.shape != (nC, nF):
                raise ValueError("Shape of supervised information matrix is inconsistent with X and y")
        tsr_matrix = get_tsr_matrix(self.supervised_4cell_matrix, self.tsr_function)

        if self.global_policy == 'ave':
            self.global_tsr_vector = np.average(tsr_matrix, axis=0)
        elif self.global_policy == 'wave':
            category_prevalences = np.asarray(y.mean(axis=0)).flatten()
            self.global_tsr_vector = np.average(tsr_matrix, axis=0, weights=category_prevalences)
        elif self.global_policy == 'sum':
            self.global_tsr_vector = np.sum(tsr_matrix, axis=0)
        elif self.global_policy == 'max':
            self.global_tsr_vector = np.amax(tsr_matrix, axis=0)
        print('fit done')

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        if not hasattr(self, 'global_tsr_vector'):
            raise NameError('TSRweighting: transform method called before fit.')
        tf_X = self.unsupervised_vectorizer.transform(X) #.toarray()
        weighted_X = csr_matrix.multiply(tf_X, self.global_tsr_vector)
        if self.norm is not None and self.norm!='none':
            weighted_X = sklearn.preprocessing.normalize(weighted_X, norm=self.norm, axis=1, copy=False)
        return scipy.sparse.csr_matrix(weighted_X)

