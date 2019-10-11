from data.dataset import init_vectorizer
from data.sentiment import fetch_MDSunprocessed
from data.tsr_function__ import get_supervised_matrix, get_tsr_matrix, posneg_information_gain, information_gain, chi_square, gss
from model.embedding_predictor import EmbeddingPredictor
from util.common import *
from sklearn.decomposition import PCA


def zscores(x, axis=0): #scipy.stats.zscores does not avoid division by 0, which can indeed occur
    std = np.clip(np.std(x, axis=axis), 1e-5, None)
    mean = np.mean(x, axis=axis)
    return (x - mean) / std

def supervised_embeddings(X,Y):
    feat_prev = (X > 0).sum(axis=0)
    F = (X.T).dot(Y) / feat_prev.T
    return F

def supervised_embeddings_tfidf(X,Y):
    tfidf_norm = X.sum(axis=0)
    F = (X.T).dot(Y) / tfidf_norm.T
    return F

def supervised_embeddings_tfidfc(X,Y):
    tfidf_norm = X.sum(axis=0)
    c_norm = Y.sum(axis=0)
    F = (X.T).dot(Y)
    F = F / tfidf_norm.T
    F = F / c_norm
    return F

def supervised_embeddings_pmi(X,Y):
    Xbin = X>0
    D = X.shape[0]
    Pxy = (Xbin.T).dot(Y)/D
    Px = Xbin.sum(axis=0)/D
    Py = Y.sum(axis=0)/D
    F = np.asarray(Pxy/(Px.T*Py))
    F = np.log1p(F)
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

def supervised_embeddings_ig(X,Y):
    def factor(pxy,px,py):
        den=px.T*py
        den[den==0]=1
        res=np.asarray(pxy/den)
        res=np.log2(res)
        res=pxy.multiply(res)
        return res

    D = X.shape[0]
    Xpos = X > 0
    Xneg = (X == 0)*1
    Yneg = (Y == 0)*1

    Px = Xpos.sum(axis=0)/D
    Pnotx = 1-Px
    Py = Y.sum(axis=0)/D
    Pnoty = 1-Py

    accum  = factor(pxy=(Xpos.T).dot(Y)/D, px=Px, py=Py)     #P(x,y)
    accum += factor(pxy=(Xneg.T).dot(Y)/D, px=Pnotx, py=Py) #P(not x, y)
    accum += factor(pxy=(Xpos.T).dot(Yneg)/D, px=Px, py=Pnoty) #P(x, not y)
    accum += factor(pxy=(Xneg.T).dot(Yneg)/D, px=Pnotx, py=Pnoty) #P( not x, not y)

    return accum.toarray()

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


# def supervised_embeddings_tsr(dataset):
#     print('computing supervised matrix with information gain')
#
#     Xtr, _ = dataset.vectorize()
#     Y = dataset.devel_labelmatrix
#     cell_matrix = get_supervised_matrix(Xtr, Y)
#     F = get_tsr_matrix(cell_matrix, tsr_score_funtion=posneg_information_gain).T
#
#     return F

def get_supervised_embeddings(X, Y, max_label_space=300, binary_structural_problems=-1, method='dotn', dozscore=True):
    print('computing supervised embeddings...')
    # tinit = time()

    nD,nF = X.shape
    nC = Y.shape[1]

    if nC==2 and binary_structural_problems > nC:
        raise ValueError('not implemented')
        # assert binary_structural_problems < nF, 'impossible to take more structural problems than features'
        # print('creating structural problems')
        # FC = get_tsr_matrix(get_supervised_matrix(X, Y), positive_information_gain).T
        # best_features_idx = np.argsort(-FC, axis=0).flatten()
        # selected=set()
        # round_robin = iter(best_features_idx)
        # while len(selected) < binary_structural_problems:
        #     selected.add(next(round_robin))
        # selected = np.asarray(list(selected))
        # structural = (X[:,selected]>0)*1
        # Y=structural

    if method=='dot':
        F = supervised_embeddings(X, Y)
    elif method=='pmi':
        F = supervised_embeddings_pmi(X, Y)
    elif method=='ppmi':
        F = supervised_embeddings_ppmi(X, Y)
    elif method == 'dotn':
        F = supervised_embeddings_tfidf(X, Y)
    elif method == 'dotc':
        F = supervised_embeddings_tfidfc(X, Y)
    elif method == 'ig':
        F = supervised_embeddings_tsr(X, Y, information_gain)
    elif method == 'pnig':
        F = supervised_embeddings_tsr(X, Y, posneg_information_gain)
    elif method == 'chi2':
        F = supervised_embeddings_tsr(X, Y, chi_square)
    elif method == 'gss':
        F = supervised_embeddings_tsr(X, Y, gss)

    if nC > max_label_space: #TODO: if predict_missing or predict_all, should it be done after of before?
        print(f'supervised matrix has more dimensions ({nC}) than the allowed limit {max_label_space}. '
              f'Applying PCA(n_components={max_label_space})')
        pca = PCA(n_components=max_label_space)
        F = pca.fit(F).transform(F)
        F /= pca.singular_values_

    if dozscore:
        F = zscores(F, axis=0)

    # print(f'took {time() - tinit}s')

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


class KeyedVectors:
    def __init__(self, word2index, weights):
        assert len(word2index)==weights.shape[0], 'wrong number of dimensions'
        index2word = {i:w for w,i in word2index.items()}
        assert len([i for i in range(len(index2word)) if i not in index2word])==0, 'gaps in indexing not allowed'
        self.word2index = word2index
        self.index2word = index2word
        self.weights = weights

    def extract(self, words):
        dim = self.weights.shape[1]
        v_size = len(words)

        source_idx, target_idx = [], []
        for i,word in enumerate(words):
            if word not in self.word2index: continue
            j = self.word2index[word]
            source_idx.append(i)
            target_idx.append(j)

        extraction = np.zeros((v_size, dim))
        extraction[np.asarray(source_idx)] = self.weights[np.asarray(target_idx)]
        return extraction


def multi_domain_sentiment_embeddings():
    data = fetch_MDSunprocessed(min_pos=1000, min_neg=1000, max_unlabeled=0)
    print('done')
    #compute the supervised embedding matrix for each domain
    vocabularies, weights = [], []
    for domain in tqdm(data.keys(), 'indexing domain:'):
        documents, labels = zip(*data[domain]['labeled'])
        labels = np.array(labels)
        tfidf = init_vectorizer()
        X = tfidf.fit_transform(documents)
        Y = np.array((1-labels, labels)).T
        F = supervised_embeddings(X, Y)
        vocabularies.append(tfidf.vocabulary_)
        weights.append(F)

    #merge supervised embedding matrices with their vocabularies
    merge_vocabuary = set()
    for v in vocabularies:
        merge_vocabuary.update(v.keys())
    merge_vocabulary_inv = dict(enumerate(merge_vocabuary))
    merge_vocabulary = {w:i for i,w in merge_vocabulary_inv.items()}
    v_size = len(merge_vocabuary)
    num_domains = len(data)
    merge_weights = np.zeros((v_size, num_domains*2))

    for d,(v,w) in tqdm(enumerate(zip(vocabularies, weights))):
        source_idx,target_idx=[],[]
        for i in range(v_size):
            word = merge_vocabulary_inv[i]
            if word not in v: continue
            j = v[word]
            source_idx.append(i)
            target_idx.append(j)
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        merge_weights[source_idx,d*2:d*2+2] = w[target_idx,:]

    return KeyedVectors(merge_vocabulary, merge_weights)


# if __name__=='__main__':
#     from scipy.sparse import csr_matrix
#
#     np.random.seed(1)
#     X = np.random.rand(100, 5)
#     X[X>0.7]=1.
#     X[X<=0.7]=0.
#     X = csr_matrix(X)
#
#     Y = np.random.rand(100, 4)
#     Y[Y>0.8]=1
#     Y[Y<=0.8]=0
#     Y = csr_matrix(Y)
#     S1 = supervised_embeddings_tsr(X, Y)
#     S2 = supervised_embeddings_ig(X, Y)
#     print(S1)
#     print(S2)
#     print(np.argsort(S1.flatten()))
#     print(np.argsort(S2.flatten()))
