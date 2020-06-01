from data.tsr_function__ import get_supervised_matrix, get_tsr_matrix, information_gain, chi_square
from model.embedding_predictor import EmbeddingPredictor
from util.common import *
from sklearn.decomposition import PCA


def zscores(x, axis=0): #scipy.stats.zscores does not avoid division by 0, which can indeed occur
    std = np.clip(np.std(x, ddof=1, axis=axis), 1e-5, None)
    mean = np.mean(x, axis=axis)
    return (x - mean) / std


def supervised_embeddings_tfidf(X,Y):
    tfidf_norm = X.sum(axis=0)
    F = (X.T).dot(Y) / tfidf_norm.T
    return F


def supervised_embeddings_ppmi(X,Y):
    Xbin = X>0
    D = X.shape[0]
    Pxy = (Xbin.T).dot(Y)/D
    Px = Xbin.sum(axis=0)/D
    Py = Y.sum(axis=0)/D
    F = np.asarray(Pxy/(Px.T*Py))
    F = np.maximum(F, 1.0)
    F = np.log(F)
    return F


def supervised_embeddings_tsr(X,Y, tsr_function=information_gain, max_documents=25000):
    D = X.shape[0]
    if D>max_documents:
        print(f'sampling {max_documents}')
        random_sample = np.random.permutation(D)[:max_documents]
        X = X[random_sample]
        Y = Y[random_sample]
    cell_matrix = get_supervised_matrix(X, Y)
    F = get_tsr_matrix(cell_matrix, tsr_score_funtion=tsr_function).T
    return F


def get_supervised_embeddings(X, Y, max_label_space=300, binary_structural_problems=-1, method='dotn', dozscore=True):
    print('computing supervised embeddings...')

    nC = Y.shape[1]
    if nC==2 and binary_structural_problems > nC:
        raise ValueError('not implemented in this branch')

    if method=='ppmi':
        F = supervised_embeddings_ppmi(X, Y)
    elif method == 'dotn':
        F = supervised_embeddings_tfidf(X, Y)
    elif method == 'ig':
        F = supervised_embeddings_tsr(X, Y, information_gain)
    elif method == 'chi2':
        F = supervised_embeddings_tsr(X, Y, chi_square)

    if dozscore:
        F = zscores(F, axis=0)

    if nC > max_label_space:
        print(f'supervised matrix has more dimensions ({nC}) than the allowed limit {max_label_space}. '
              f'Applying PCA(n_components={max_label_space})')
        pca = PCA(n_components=max_label_space)
        F = pca.fit(F).transform(F)

    return F


def fit_predict(W, F, mode='all'):
    """
    learns a regression function from W->F using the embeddings which appear both in W and F and produces a prediction
    for the missing embeddings in F (mode='missing') or for all (mode='all')
    :param W: an embedding matrix of shape (V1,E1), i.e., vocabulary-size-1 x embedding-size-1
    :param F: an embedding matris of shape (V2,E2), i.e., vocabulary-size-2 x embedding-size-2
    :param mode: indicates which embeddings are to be predicted. mode='all' will return a matrix of shape (V1,E2) where
    all V1 embeddings are predicted; when mode='missing' only the last V1-V2 embeddings will be predicted, and the previous
    V2 embeddings are copied
    :return: an embedding matrix of shape (V1,E2)
    """
    V1,E1=W.shape
    V2,E2=F.shape
    assert mode in {'all','missing'}, 'wrong mode; availables are "all" or "missing"'

    e = EmbeddingPredictor(input_size=E1, output_size=E2).cuda()
    e.fit(W[:V2], F)
    if mode == 'all':
        print('learning to predict all embeddings')
        F = e.predict(W)
    elif mode=='missing':
        print('learning to predict only the missing embeddings')
        Fmissing = e.predict(W[V2:])
        F = np.vstack((F, Fmissing))
    return F





