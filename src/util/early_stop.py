#adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import torch
from time import time
from util.file import create_if_not_exist


class EarlyStopping:

    def __init__(self, model, patience=20, verbose=True, checkpoint='./checkpoint.pt'):
        # set patience to 0 or -1 to avoid stopping, but still keeping track of the best value and model parameters
        self.patience_limit = patience
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.best_epoch = None
        self.stop_time  = None
        self.checkpoint = checkpoint
        self.model = model
        self.STOP = False

    def __call__(self, watch_score, epoch):

        if self.STOP:
            return #done

        if self.best_score is None or watch_score >= self.best_score:
            self.best_score = watch_score
            self.best_epoch = epoch
            self.stop_time = time()
            if self.checkpoint:
                self.print(f'[early-stop] improved, saving model in {self.checkpoint}')
                torch.save(self.model, self.checkpoint)
            else:
                self.print(f'[early-stop] improved')
            self.patience = self.patience_limit
        else:
            self.patience -= 1
            if self.patience == 0:
                self.STOP = True
                self.print(f'[early-stop] patience exhausted')
            else:
                if self.patience>0: # if negative, then early-stop is ignored
                    self.print(f'[early-stop] patience={self.patience}')

    def reinit_counter(self):
        self.STOP = False
        self.patience=self.patience_limit

    def restore_checkpoint(self):
        return torch.load(self.checkpoint)

    def print(self, msg):
        if self.verbose:
            print(msg)
