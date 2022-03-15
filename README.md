# CS4248-Project

### Demo
I found related tutorial online and this demo reproduces the work of https://aclanthology.org/D16-1058.pdf. The architecture of this model has been put in  `architecture` folder.

In order to run the demo successfully, you have to download the pre-trained word vectors, which is available at https://nlp.stanford.edu/projects/glove/. I used `glove.6B.300d.txt` in this demo, remember to decompress this text file into the `data` folder.

### Codes for ATAE-LSTM (Two subclasses: `Attention` and `Final_pred`)
![image.png](https://cdn.nlark.com/yuque/0/2022/png/2896592/1647354744981-e5100037-9bed-49e9-9aba-276f16915410.png#clientId=ua89ebd79-5fab-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=747&id=ub47bb506&margin=%5Bobject%20Object%5D&name=image.png&originHeight=747&originWidth=1595&originalType=binary&ratio=1&rotation=0&showTitle=false&size=158965&status=done&style=none&taskId=uedccfdfc-4788-440b-94ca-78268c7cb5b&title=&width=1595)
```python
class ATAE_LSTM(nn.Module):
    def __init__(self, ):
        self.lstm = nn.LSTM(input_size=2 * embedding_dim,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=True)
    
    def forward(self,):
          e1 = self.text_embeddings(text)
          # e1 shape(batch_size,seq_len, embedding_dim)
          e2 = self.term_embeddings(term).expand(e1.size())

          wv = torch.cat((e1, e2), dim=2)
          # e.g.
          # wv torch.Size([batch_size,seq_len,2*embedding_dim])

          out, (h, c) = self.lstm(wv)  # output, (h, c)
          # out shape(batch_size,seq_len, 2 * num_hiddens)
          # h shape(num_layers * num_directions, batch_size, 2*num_hiddens)
```
![image.png](https://cdn.nlark.com/yuque/0/2022/png/2896592/1647354917886-0c0113af-18cf-4054-90d7-537ad4b5e9d5.png#clientId=ua89ebd79-5fab-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=615&id=u3ce18ba5&margin=%5Bobject%20Object%5D&name=image.png&originHeight=615&originWidth=1639&originalType=binary&ratio=1&rotation=0&showTitle=false&size=147471&status=done&style=none&taskId=ud7c26c56-b7de-4efa-bc4a-9878bb91adb&title=&width=1639)
```python
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

class ATAE_LSTM(nn.Module):
    def __init__(self, ):
        self.attn = Attention_mlp(embedding_dim,2 * num_hiddens)
    
    def forward(self,):
        r = self.attn(self.term_embeddings(term), out)
```
![image.png](https://cdn.nlark.com/yuque/0/2022/png/2896592/1647354986441-68c5e0b6-53cf-4a57-b2b8-4faa9b5b3048.png#clientId=ua89ebd79-5fab-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=363&id=ufa357120&margin=%5Bobject%20Object%5D&name=image.png&originHeight=363&originWidth=1684&originalType=binary&ratio=1&rotation=0&showTitle=false&size=50409&status=done&style=none&taskId=u50540efa-948a-4750-b131-f1c73cd5dfe&title=&width=1684)
```python
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
    def __init__(self, ):
        self.final_pred = Final_pred(2 *num_hiddens)

    
    def forward(self,):
        h_n = out[:, -1:, :]
        # shape(batch_size * 1 * hidden_dim)
        y = self.final_pred(r, h_n)
        # shape(batch_size * 1 * 3)
        return y
```

