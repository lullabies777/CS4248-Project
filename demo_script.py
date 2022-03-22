# %%
# import packages
from DeBERTa_model.deberta import DeBERTa
from DeBERTa_model.sequence_classification import SequenceClassificationModel
from transformers import DebertaV2Model
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import pandas as pd 
import os
from tqdm import tqdm
from torchtext.legacy import data
from torchtext.vocab import Vectors
from torchtext.vocab import GloVe
import time
import math

# %%
# preprocessing

from nltk import word_tokenize

text = data.Field(sequential = True, lower = True, tokenize = word_tokenize)
term = data.Field(sequential = False, lower = True)
polarity = data.Field(sequential = False)

# %%
train, val = data.TabularDataset.splits(path=r'data/',
                                        skip_header=True,
                                        train='rest_train.csv',
                                        validation='rest_test.csv',
                                        format='csv',
                                        fields=[('text', text),
                                                ('term', term),
                                                ('polarity', polarity)])

# %%
vectors = Vectors(name='data/glove.6B.300d.txt')

# %%
text.build_vocab(train, val, vectors=vectors)
term.build_vocab(train, val, vectors=vectors)
polarity.build_vocab(train, val)

text_vocab_size = len(text.vocab)
term_vocab_size = len(term.vocab)
text_vector=text.vocab.vectors
term_vector=term.vocab.vectors

# %%
# batch_size=512
batch_size=16
train_iter, val_iter = data.Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(val)), # batch_size only for training
    )   

# %%
class Attention_mlp(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Attention_mlp, self).__init__()
        self.wv = nn.Linear(embedding_dim, embedding_dim, bias= False)
        self.wh = nn.Linear(hidden_dim, embedding_dim, bias = False)
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(2 * embedding_dim, 1, bias = False)
    def forward(self,term, hidden):
        #### term shape: batch_size * 1 * embedding
        #### hidden shape: batch_size * seq_len * hidden_dim
        term1 = self.wv(term).transpose(-2,-1)
        # shape(batch_size * embedding_dim * 1)
        hidden1 = self.wh(hidden).transpose(-2,-1)
        # shape(batch_size * embedding_dim * seq_len)

        M = torch.cat((hidden1, term1.expand(hidden1.size())), dim = -2)
        # shape(batch_size * (2 * embedding_dim) * seq_len)

        alpha = F.softmax(self.fc1(torch.tanh(M.transpose(-2,-1))), dim = -2).transpose(-2,-1)
        # shape(batch_size * 1 * seq_len)
        
        h_star = torch.matmul(alpha, hidden)
        # shape(batch_size * 1 * hidden_dim)
        return h_star

class Final_pred(nn.Module):
    def __init__(self, hidden_dim):
        super(Final_pred, self).__init__()
        self.wp = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.wx = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.ws = nn.Linear(hidden_dim, 3)

    def forward(self, h_star, h_n):
        o_star = torch.tanh(self.wp(h_star) + self.wx(h_n))
        # shape(batch_size * 1 * hidden_dim)
        y = self.ws(o_star)
        # shape(batch_size * 1 * 3)
        return y.squeeze(1)
        
class ATAE_LSTM(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers):
        super(ATAE_LSTM, self).__init__()
        self.text_embeddings = nn.Embedding(text_vocab_size, embedding_dim)
        self.term_embeddings = nn.Embedding(term_vocab_size, embedding_dim)
        self.text_embeddings = nn.Embedding.from_pretrained(text_vector,
                                                            freeze=False)
        self.term_embeddings = nn.Embedding.from_pretrained(term_vector,
                                                              freeze=False)
        self.lstm = nn.LSTM(input_size=2 * embedding_dim,
                            hidden_size=num_hiddens,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        # self.wp = nn.Parameter(torch.Tensor(num_hiddens * 2, num_hiddens * 2))
        # self.wx = nn.Parameter(torch.Tensor(num_hiddens * 2, num_hiddens * 2))
        # self.ws = nn.Parameter(torch.Tensor(3, num_hiddens * 2))
        
        self.attn = Attention_mlp(embedding_dim,2 * num_hiddens)

        self.final_pred = Final_pred(2 *num_hiddens)

    def forward(self, text, term):
        seq_len = len(text.t())
        # print('text2:',text.size(1))
        # print('term:',term.size())
        e1 = self.text_embeddings(text)
        # e1 shape(batch_size,seq_len, embedding_dim)
        e2 = self.term_embeddings(term).expand(e1.size())

        wv = torch.cat((e1, e2), dim=2)
        # e.g.
        # wv torch.Size([batch_size,seq_len,2*embedding_dim])

        out, (h, c) = self.lstm(wv)  # output, (h, c)
        # out shape(batch_size,seq_len, 2 * num_hiddens)
        # h shape(num_layers * num_directions, batch_size, 2*num_hiddens)

        r = self.attn(self.term_embeddings(term), out)
        # shape(batch_size * 1 * hidden_dim)
        h_n = out[:, -1:, :]
        # shape(batch_size * 1 * hidden_dim)
        y = self.final_pred(r, h_n)
        # shape(batch_size * 1 * 3)
        return y

# %%
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X1, X2, y = batch.text.cuda(), batch.term.cuda(), batch.polarity.cuda()
            X1 = X1.permute(1, 0)
            X2 = X2.unsqueeze(1)
            y.data.sub_(1)  # index start from 0
            if isinstance(net, torch.nn.Module):
                net.eval()  
                acc_sum += (net(X1,
                                X2).argmax(dim=1) == y).float().sum().item()
                net.train()  
            else:
                if ('is_training'
                        in net.__code__.co_varnames): 
                    acc_sum += (net(X1, X2, is_training=False).argmax(
                        dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(
                        X1, X2).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# %%
def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch_idx, batch in enumerate(train_iter):
            X1, X2, y = batch.text.cuda(), batch.term.cuda(), batch.polarity.cuda()
            X1 = X1.permute(1, 0).cuda()
            X2 = X2.unsqueeze(1).cuda()
            y.data.sub_(1)  # index start from 0
            y_hat = net(X1,X2)
            # y_hat = y_hat.logits
            # y_hat = torch.argmax(y_hat, dim=-1)
            print(y_hat[0].shape, y.shape, X1.shape, X2.shape)
            l = loss(y_hat[0], y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))

# %%
embedding_dim, num_hiddens, num_layers = 300, 150, 1
# net = ATAE_LSTM(embedding_dim, num_hiddens, num_layers).cuda()
cur_dir_path = os.path.dirname(os.path.abspath(__file__)) # Added
model_path = os.path.join(cur_dir_path, 'DeBERTa_model', 'base') # Added
# net = SequenceClassificationModel(pre_trained=model_path).cuda() # Added
net = DebertaV2Model.from_pretrained(os.path.join(model_path, 'pytorch.model.bin'), config=os.path.join(model_path, 'model_config.json')).cuda()
print(net)
lr, num_epochs = 0.001, 20
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, val_iter, net, loss, optimizer, num_epochs)


