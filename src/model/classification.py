from model.layers import *
from transformers import BertModel, BertTokenizer
import logging

logging.basicConfig(level=logging.INFO)


class NeuralClassifier(nn.Module):
    ALLOWED_NETS = {'cnn', 'lstm', 'attn'}

    def __init__(self,
                 net_type,
                 output_size,
                 hidden_size,
                 vocab_size,
                 learnable_length,
                 pretrained=None,
                 drop_embedding_range=None,
                 drop_embedding_prop=0):
        super(NeuralClassifier, self).__init__()

        self.embed = EmbeddingCustom(vocab_size, learnable_length, pretrained, drop_embedding_range,
                                     drop_embedding_prop)
        self.projection = init__projection(net_type)(self.embed.dim(), hidden_size)
        self.label = nn.Linear(self.projection.dim(), output_size)

    def forward(self, input):
        word_emb = self.embed(input)
        doc_emb = self.projection(word_emb)
        logits = self.label(doc_emb)
        return logits

    def finetune_pretrained(self):
        self.embed.finetune_pretrained()

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)


class Token2BertEmbeddings:
    def __init__(self, pretrained_model_name='bert-base-uncased', max_length=500, device='cuda'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name).eval().to(device)
        self.max_length = max_length
        self.device = device

    def embeddings(self, tokens):
        max_length = min(self.max_length, max(map(len, tokens)))  # for dynamic padding
        cls_t = self.tokenizer.cls_token
        sep_t = self.tokenizer.sep_token
        pad_idx = self.tokenizer.pad_token_id
        tokens = [[cls_t] + d[:max_length] + [sep_t] for d in tokens]
        index = [
            self.tokenizer.convert_tokens_to_ids(doc) + [pad_idx] * (max_length - (len(doc) - 2)) for doc in
            tokens
        ]
        # index = [
        #    self.tokenizer.encode(d, add_special_tokens=True, max_length=max_length+2, pad_to_max_length=True)
        #    for d in docs
        # ]
        index = torch.tensor(index).to(self.device)

        with torch.no_grad():
            outputs = self.model(index)
            contextualized_embeddings = outputs[0]
            # ignore embeddings for [CLS] and las one (either [SEP] or last [PAD])
            contextualized_embeddings = contextualized_embeddings[:, 1:-1, :]
            return contextualized_embeddings

    def dim(self):
        return 768


class Token2WCEmbeddings(nn.Module):
    def __init__(self, WCE, WCE_range, WCE_vocab, drop_embedding_prop=0.5, max_length=500, device='cuda'):
        super(Token2WCEmbeddings, self).__init__()
        assert '[PAD]' in WCE_vocab, 'unknown index for special token [PAD] in WCE vocabulary'
        self.embed = EmbeddingCustom(len(WCE_vocab), 0, WCE, WCE_range, drop_embedding_prop).to(device)
        self.max_length = max_length
        self.device = device
        self.vocab = WCE_vocab
        self.pad_idx = self.vocab['[PAD]']
        self.unk_idx = self.vocab['[UNK]']

    def forward(self, tokens):
        max_length = min(self.max_length, max(map(len, tokens)))  # for dynamic padding
        tokens = [d[:max_length] for d in tokens]
        index = [
            [self.vocab.get(ti, self.unk_idx) for ti in doc] + [self.pad_idx] * (max_length - len(doc)) for doc in
            tokens
        ]
        index = torch.tensor(index).to(self.device)
        return self.embed(index)

    def dim(self):
        return self.embed.dim()

    def finetune_pretrained(self):
        self.embed.finetune_pretrained()


class BertWCEClassifier(nn.Module):
    ALLOWED_NETS = {'cnn', 'lstm', 'attn'}

    def __init__(self,
                 net_type,
                 output_size,
                 hidden_size,
                 token2bert_embeddings,
                 token2wce_embeddings):
        super(BertWCEClassifier, self).__init__()

        emb_dim = token2bert_embeddings.dim() + (0 if token2wce_embeddings is None else token2wce_embeddings.dim())
        print(f'Embedding dimensions {emb_dim}')

        self.token2bert_embeddings = token2bert_embeddings
        self.token2wce_embeddings = token2wce_embeddings
        self.projection = init__projection(net_type)(emb_dim, hidden_size)
        self.label = nn.Linear(self.projection.dim(), output_size)

    def forward(self, input):  # list of lists of tokens
        # convert tokens to id for Bert, pad, and get contextualized embeddings
        contextualized_embeddings = self.token2bert_embeddings.embeddings(input)

        # convert tokens to ids for WCE, pad, and get WCEs
        if self.token2wce_embeddings is not None:
            wce_embeddings = self.token2wce_embeddings(input)
            # concatenate Bert embeddings with WCEs
            assert contextualized_embeddings.shape[1] == wce_embeddings.shape[1], 'shape mismatch between Bert and WCE'
            word_emb = torch.cat([contextualized_embeddings, wce_embeddings], dim=-1)
        else:
            word_emb = contextualized_embeddings

        doc_emb = self.projection(word_emb)
        logits = self.label(doc_emb)
        return logits

    def finetune_pretrained(self):
        self.token2wce_embeddings.finetune_pretrained()

    def xavier_uniform(self):
        for model in [self.token2wce_embeddings, self.projection, self.label]:
            if model is None: continue
            for p in model.parameters():
                if p.dim() > 1 and p.requires_grad:
                    nn.init.xavier_uniform_(p)


def init__projection(net_type):
    assert net_type in NeuralClassifier.ALLOWED_NETS, 'unknown network'
    if net_type == 'cnn':
        return CNNprojection
    elif net_type == 'lstm':
        return LSTMprojection
    elif net_type == 'attn':
        return ATTNprojection


class BertClassifier(nn.Module):

    def __init__(self, output_size, pretrained_model_name='bert-base-uncased', max_length=500, dropout=0.1,
                 device='cuda'):
        super(BertClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name).eval().to(device)
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.dim(), output_size)
        self.max_length = max_length
        self.device = device

    def cls_embedding_from_bert_model(self, tokens):
        max_length = min(self.max_length, max(map(len, tokens)))  # for dynamic padding
        cls_t = self.tokenizer.cls_token
        sep_t = self.tokenizer.sep_token
        pad_idx = self.tokenizer.pad_token_id
        tokens = [[cls_t] + d[:max_length] + [sep_t] for d in tokens]
        index = [
            self.tokenizer.convert_tokens_to_ids(doc) + [pad_idx] * (max_length - (len(doc) - 2)) for doc in
            tokens
        ]
        # index = [
        #    self.tokenizer.encode(d, add_special_tokens=True, max_length=max_length+2, pad_to_max_length=True)
        #    for d in docs
        # ]
        index = torch.tensor(index).to(self.device)

        outputs = self.model(index)
        # get [CLS] embedding
        return outputs[0][:, 0, :]

    def dim(self):
        return 768

    def forward(self, input):  # list of lists of tokens
        cls_embedding = self.cls_embedding_from_bert_model(input)
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classification(cls_embedding)
        return logits

    def xavier_uniform(self):
        for model in [self.classification]:
            if model is None: continue
            for p in model.parameters():
                if p.dim() > 1 and p.requires_grad:
                    nn.init.xavier_uniform_(p)
