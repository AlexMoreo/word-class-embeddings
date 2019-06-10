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

    def fit(self, U, S):
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.parameters())
        batchsize = 100
        nepochs = 500

        early_stop = EarlyStopping(self, patience=20, checkpoint=None, verbose=True)

        epoch=0
        while not early_stop.STOP and epoch<nepochs:
            nbatches = U.shape[0]//batchsize
            nbatches += 1 if U.shape[0] % batchsize > 0 else 0
            losses = []
            for i in range(nbatches):
                batch_u = U[i * batchsize:(i + 1) * batchsize].cuda()
                batch_s = torch.FloatTensor(S[i * batchsize:(i + 1) * batchsize]).cuda()
                optim.zero_grad()
                output = self.forward(batch_u)
                loss = criterion(output, batch_s)
                loss.backward()
                optim.step()
                loss_value = loss.item()
                losses.append(loss_value)

            loss_mean = np.mean(losses)
            loss_std  = np.std(losses)
            print(f'epoch {epoch}: loss={loss_mean:.5f} +-{loss_std:.5f}')
            if epoch>5:
                early_stop(-loss_mean, epoch)
            epoch+=1

    def predict(self, U):
        U = U.cuda()
        return self.forward(U).detach().cpu().numpy() #todo: batchify
        # return F.relu(self.lin1(U)).detach().cpu().numpy()


