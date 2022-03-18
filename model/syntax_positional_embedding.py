import torch.nn as nn
import torch

class SyntaxPositionalEmbedding(nn.Module):

    def __init__(self,
                 hidden_dimension,
                 d_vocab,
                 c_vocab,
                 u_vocab,
                 d_dim,
                 c_dim,
                 u_dim,
                 padding_idx=0):
        self.hidden_dimension = hidden_dimension
        self.d_vocab = d_vocab
        self.c_vocab = c_vocab
        self.u_vocab = u_vocab
        self.embedding_d = nn.Embedding(d_vocab, d_dim, padding_idx)
        self.embedding_c = nn.Embedding(c_vocab, c_dim, padding_idx)
        self.embedding_u = nn.Embedding(u_vocab, u_dim, padding_idx)

    def forward(self, seqs, d, c, u):
        d = self.embedding_d(d) #BxL-1xd
        c = self.embedding_c(c) #BxL-1xd
        u = self.embedding_u(u) #BxLxd

        seqs_u = seqs + u
        d_c = torch.cat((d, c), dim=2)





