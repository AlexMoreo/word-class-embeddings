import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, issparse
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics.classification import _check_set_wise_labels, multilabel_confusion_matrix, _prf_divide

"""
Scikit learn provides a full set of evaluation metrics, but they treat special cases differently.
I.e., when the number of true positives, false positives, and false negatives ammount to 0, all
affected metrices (precision, recall, and thus f1) output 0 in Scikit learn.
We adhere to the common practice of outputting 1 in this case since the classifier has correctly
classified all examples as negatives.
"""

def evaluation(y_true, y_pred, classification_type):
    if classification_type == 'multilabel':
        Mf1,mf1,accuracy = multilabel_eval(y_true, y_pred)
        # Mf1 = f1_score(y_true, y_pred, average='macro') #TODO: repair
        # p,r,f1,support=precision_recall_fscore_support(y_true, y_pred)
        # f1[support==0]
        # Mf1 = macro_f1_repaired(y_true, y_pred)
        # mf1 = f1_score(y_true, y_pred, average='micro')
        # accuracy = accuracy_score(y_true, y_pred)
    elif classification_type == 'singlelabel':
        if issparse(y_pred): y_pred=y_pred.toarray().flatten()
        Mf1 = f1_score(y_true, y_pred, average='macro')
        mf1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)

    return Mf1, mf1, accuracy

def multilabel_eval(y, y_):

    tp = y.multiply(y_)

    fn = lil_matrix(y.shape)
    true_ones = y==1
    fn[true_ones]=1-tp[true_ones]

    fp = lil_matrix(y.shape)
    pred_ones = y_==1
    if pred_ones.nnz>0:
        fp[pred_ones]=1-tp[pred_ones]

    #macro-f1
    tp_macro = np.asarray(tp.sum(axis=0), dtype=int).flatten()
    fn_macro = np.asarray(fn.sum(axis=0), dtype=int).flatten()
    fp_macro = np.asarray(fp.sum(axis=0), dtype=int).flatten()

    pos_pred = tp_macro+fp_macro
    pos_true = tp_macro+fn_macro
    prec=np.zeros(shape=tp_macro.shape,dtype=float)
    rec=np.zeros(shape=tp_macro.shape,dtype=float)
    np.divide(tp_macro, pos_pred, out=prec, where=pos_pred>0)
    np.divide(tp_macro, pos_true, out=rec, where=pos_true>0)
    den=prec+rec

    macrof1=np.zeros(shape=tp_macro.shape,dtype=float)
    np.divide(np.multiply(prec,rec),den,out=macrof1,where=den>0)
    macrof1 *=2

    macrof1[(pos_pred==0)*(pos_true==0)]=1
    macrof1 = np.mean(macrof1)

    #micro-f1
    tp_micro = tp_macro.sum()
    fn_micro = fn_macro.sum()
    fp_micro = fp_macro.sum()
    pos_pred = tp_micro + fp_micro
    pos_true = tp_micro + fn_micro
    prec = (tp_micro / pos_pred) if pos_pred>0 else 0
    rec  = (tp_micro / pos_true) if pos_true>0 else 0
    den = prec+rec
    microf1 = 2*prec*rec/den if den>0 else 0
    if pos_pred==pos_true==0:
        microf1=1

    #accuracy
    ndecisions = np.multiply(*y.shape)
    tn = ndecisions - (tp_micro+fn_micro+fp_micro)
    acc = (tp_micro+tn)/ndecisions

    return macrof1,microf1,acc



# def macro_f1_repaired(y_true, y_pred):
#     beta = 1.0
#     labels = None,
#     pos_label = 1
#     average='macro'
#     sample_weight = None
#     warn_for = ('f-score',)
#
#     labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
#
#     # Calculate tp_sum, pred_sum, true_sum ###
#     samplewise = average == 'samples'
#     MCM = multilabel_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels, samplewise=samplewise)
#     tp_sum = MCM[:, 1, 1]
#     pred_sum = tp_sum + MCM[:, 0, 1]
#     true_sum = tp_sum + MCM[:, 1, 0]
#
#     # Finally, we have all our sufficient statistics. Divide! #
#     beta2 = beta ** 2
#
#     # Divide, and on zero-division, set scores to 0 and warn:
#     precision = _prf_divide(tp_sum, pred_sum, 'precision', 'predicted', average, warn_for)
#     recall = _prf_divide(tp_sum, true_sum, 'recall', 'true', average, warn_for)
#     # Don't need to warn for F: either P or R warned, or tp == 0 where pos
#     # and true are nonzero, in which case, F is well-defined and zero
#     denom = beta2 * precision + recall
#     denom[denom == 0.] = 1  # avoid division by 0
#     f_score = (1 + beta2) * precision * recall / denom
#
#     f_score[(pred_sum==0)*(true_sum==0)]=1
#     f_score = np.mean(f_score)
#
#     return f_score


