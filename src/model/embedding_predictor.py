from torch.nn import functional as F
from model.helpers import *
import numpy as np

from util.early_stop import EarlyStopping





class EmbeddingPredictor(nn.Module):
    def __init__(self, input_size, output_size, hiddensize=64):
        super(EmbeddingPredictor, self).__init__()
        self.lin1 = nn.Linear(input_size, hiddensize)
        self.lin2 = nn.Linear(hiddensize, output_size)

    def forward(self, input):
        h = self.lin1(input)
        h = F.relu(h)
        o = self.lin2(h)
        return o

    def fit(self, U, S, val_prop=0.2):
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.parameters())
        batchsize = 100
        nepochs = 5000
        n = U.shape[0]
        indexes = np.arange(n)
        indexes = np.random.permutation(indexes)
        val_index, tr_index = indexes[:int(n*val_prop)], indexes[int(n*val_prop):]
        vU,vS = U[val_index], S[val_index]
        U,S = U[tr_index], S[tr_index]

        print(f'[learning to predict word-class embeddings from trU={U.shape}] and vaU={vU.shape}')
        early_stop = EarlyStopping(self, patience=10, checkpoint=None, verbose=True)

        epoch=0
        while not early_stop.STOP and epoch<nepochs:
            tr_losses = []
            self.train()
            for batch_u, batch_s in self.batchify(U, S, batchsize):
                optim.zero_grad()
                output = self.forward(batch_u)
                loss = criterion(output, batch_s)
                loss.backward()
                optim.step()
                loss_value = loss.item()
                tr_losses.append(loss_value)

            self.eval()
            va_losses = []
            for batch_u, batch_s in self.batchify(vU, vS, batchsize):
                output = self.forward(batch_u)
                loss = criterion(output, batch_s)
                loss_value = loss.item()
                va_losses.append(loss_value)

            tr_loss_mean, tr_loss_std = np.mean(tr_losses), np.std(tr_losses)
            va_loss_mean, va_loss_std = np.mean(va_losses), np.std(va_losses)
            print(f'epoch {epoch}: tr_loss={tr_loss_mean:.5f} +-{tr_loss_std:.5f}\t va_loss={va_loss_mean:.5f} +-{va_loss_std:.5f}')
            if epoch>5:
                early_stop(-va_loss_mean, epoch)
            epoch+=1

    def batchify(self, U, S, batchsize):
        size = U.shape[0]
        nbatches_ = size // batchsize
        nbatches_ += 1 if size % batchsize > 0 else 0
        for i in range(nbatches_):
            batch_u = U[i * batchsize:(i + 1) * batchsize].cuda()
            batch_s = torch.FloatTensor(S[i * batchsize:(i + 1) * batchsize]).cuda()
            yield batch_u, batch_s

    def predict(self, U):
        U = U.cuda()
        return self.forward(U).detach().cpu().numpy() #todo: batchify
        # return F.relu(self.lin1(U)).detach().cpu().numpy()


