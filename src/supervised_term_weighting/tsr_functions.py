import math
import numpy as np
from scipy.stats import t
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm



def positive_information_gain(cell):
    if cell.tpr() < cell.fpr():
        return 0.0
    else:
        return information_gain(cell)


def posneg_information_gain(cell):
    ig = information_gain(cell)
    if cell.tpr() < cell.fpr():
        return -ig
    else:
        return ig


def __ig_factor(p_tc, p_t, p_c):
    den = p_t * p_c
    if den != 0.0 and p_tc != 0:
        return p_tc * math.log(p_tc / den, 2)
    else:
        return 0.0


def information_gain(cell):
    return __ig_factor(cell.p_tp(), cell.p_f(), cell.p_c()) + \
           __ig_factor(cell.p_fp(), cell.p_f(), cell.p_not_c()) +\
           __ig_factor(cell.p_fn(), cell.p_not_f(), cell.p_c()) + \
           __ig_factor(cell.p_tn(), cell.p_not_f(), cell.p_not_c())


def pointwise_mutual_information(cell):
    return __ig_factor(cell.p_tp(), cell.p_f(), cell.p_c())


def gain_ratio(cell):
    pc = cell.p_c()
    pnc = 1.0 - pc
    norm = pc * math.log(pc, 2) + pnc * math.log(pnc, 2)
    return information_gain(cell) / (-norm)


def chi_square(cell):
    den = cell.p_f() * cell.p_not_f() * cell.p_c() * cell.p_not_c()
    if den==0.0: return 0.0
    num = gss(cell)**2
    return num / den


def relevance_frequency(cell):
    a = cell.tp
    c = cell.fp
    if c == 0:
        c = 1.
    return math.log(2.0 + (a * 1.0 / c), 2)


def idf(cell):
    if cell.p_f() > 0:
        return math.log(1.0 / cell.p_f())

    return 0.0


def gss(cell):
    return cell.p_tp()*cell.p_tn() - cell.p_fp()*cell.p_fn()


#set cancel_features=True to allow some features to be weighted as 0 (as in the original article)
#however, for some extremely imbalanced dataset caused all documents to be 0
def conf_weight(cell, cancel_features=False):
    def conf_interval(xt, n):
        if n > 30:
            z2 = 3.84145882069  # norm.ppf(0.5+0.95/2.0)**2
        else:
            z2 = t.ppf(0.5 + 0.95 / 2.0, df=max(n - 1, 1)) ** 2
        p = (xt + 0.5 * z2) / (n + z2)
        amplitude = 0.5 * z2 * math.sqrt((p * (1.0 - p)) / (n + z2))
        return p, amplitude

    def strength(minPosRelFreq, minPos, maxNeg):
        if minPos > maxNeg:
            return math.log(2.0 * minPosRelFreq, 2.0)
        else:
            return 0.0

    c = cell.get_c()
    not_c = cell.get_not_c()
    tp = cell.tp
    fp = cell.fp

    pos_p, pos_amp = conf_interval(tp, c)
    neg_p, neg_amp = conf_interval(fp, not_c)

    min_pos = pos_p-pos_amp
    max_neg = neg_p+neg_amp
    den = (min_pos + max_neg)
    minpos_relfreq = min_pos / (den if den != 0 else 1)

    str_tplus = strength(minpos_relfreq, min_pos, max_neg);

    if str_tplus == 0 and not cancel_features:
        return 1e-20

    return str_tplus


class ContTable:
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def get_d(self):
        return self.tp + self.tn + self.fp + self.fn

    def get_c(self):
        return self.tp + self.fn

    def get_not_c(self):
        return self.tn + self.fp

    def get_f(self):
        return self.tp + self.fp

    def get_not_f(self):
        return self.tn + self.fn

    def p_c(self):
        return (1.0*self.get_c())/self.get_d()

    def p_not_c(self):
        return 1.0-self.p_c()

    def p_f(self):
        return (1.0*self.get_f())/self.get_d()

    def p_not_f(self):
        return 1.0-self.p_f()

    def p_tp(self):
        return (1.0*self.tp) / self.get_d()

    def p_tn(self):
        return (1.0*self.tn) / self.get_d()

    def p_fp(self):
        return (1.0*self.fp) / self.get_d()

    def p_fn(self):
        return (1.0*self.fn) / self.get_d()

    def tpr(self):
        c = 1.0*self.get_c()
        return self.tp / c if c > 0.0 else 0.0

    def fpr(self):
        _c = 1.0*self.get_not_c()
        return self.fp / _c if _c > 0.0 else 0.0


def feature_label_contingency_table(positive_document_indexes, feature_document_indexes, nD):
    tp_ = len(positive_document_indexes & feature_document_indexes)
    fp_ = len(feature_document_indexes - positive_document_indexes)
    fn_ = len(positive_document_indexes - feature_document_indexes)
    tn_ = nD - (tp_ + fp_ + fn_)
    return ContTable(tp=tp_, tn=tn_, fp=fp_, fn=fn_)


def category_tables(feature_sets, category_sets, c, nD, nF):
    return [
        feature_label_contingency_table(category_sets[c], feature_sets[f], nD) for f in range(nF)
    ]


"""
Computes the nC x nF supervised matrix M where Mcf is the 4-cell contingency table for feature f and class c.
Efficiency O(nF x nC x log(S)) where S is the sparse factor
"""
def get_supervised_matrix(coocurrence_matrix, label_matrix, n_jobs=-1):
    nD, nF = coocurrence_matrix.shape
    nD2, nC = label_matrix.shape

    if nC==1:
        raise ValueError('supervised matrix has to be in binary multiclass format')
    if nD != nD2:
        raise ValueError('Number of rows in coocurrence matrix shape %s and label matrix shape %s is not consistent' %
                         (coocurrence_matrix.shape,label_matrix.shape))

    def nonzero_set(matrix, col):
        return set(matrix[:, col].nonzero()[0])

    if isinstance(coocurrence_matrix, csr_matrix):
        coocurrence_matrix = csc_matrix(coocurrence_matrix)
    feature_sets = [nonzero_set(coocurrence_matrix, f) for f in tqdm(list(range(nF)), desc='indexing features')]
    category_sets = [nonzero_set(label_matrix, c) for c in tqdm(list(range(nC)), desc='indexing categories')]
    cell_matrix = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(category_tables)(feature_sets, category_sets, c, nD, nF) for c in range(nC)
    )
    print('supervised cell matrix created')
    return np.array(cell_matrix)


# obtains the matrix T where Tcf=tsr(f,c) is the tsr score for category c and feature f
def get_tsr_matrix(cell_matrix, tsr_score_function, n_jobs=-1):
    nC = len(cell_matrix)
    nF = len(cell_matrix[0])
    tsr_matrix = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(category_tsr)(tsr_score_function, cell_matrix, c, nF) for c in range(nC)
    )
    print('TSR matrix [done]')
    return np.array(tsr_matrix)


def category_tsr(tsr_score_function, cell_matrix, c, nF):
    return [tsr_score_function(cell_matrix[c,f]) for f in range(nF)]

""" The Fisher-score [1] is not computed on the 4-cell contingency table, but can
take as input any real-valued feature column (e.g., tf-idf weights).
feat is the feature vector, and c is a binary classification vector.
This implementation covers only the binary case, while the formula is defined for multiclass
single-label scenarios, for which the version [2] might be preferred.
[1] R.O. Duda, P.E. Hart, and D.G. Stork. Pattern classification. Wiley-interscience, 2012.
[2] Gu, Q., Li, Z., & Han, J. (2012). Generalized fisher score for feature selection. arXiv preprint arXiv:1202.3725.
"""
def fisher_score_binary(feat, c):
    neg = np.ones_like(c) - c

    npos = np.sum(c)
    nneg = np.sum(neg)

    mupos = np.mean(feat[c == 1])
    muneg = np.mean(feat[neg == 1])
    mu = np.mean(feat)

    stdpos = np.std(feat[c == 1])
    stdneg = np.std(feat[neg == 1])

    num = npos * ((mupos - mu) ** 2) + nneg * ((muneg - mu) ** 2)
    den = npos * (stdpos ** 2) + nneg * (stdneg ** 2)

    if den>0:
        return num / den
    else:
        return num
