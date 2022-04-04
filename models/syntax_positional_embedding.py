import torch.nn as nn
import torch


class SimpleSyntaxPositionalEncoding(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        d_vocab_size=1000,
        c_vocab_size=155,
        u_vocab_size=27,
    ):
        super(SimpleSyntaxPositionalEncoding, self).__init__()
        self.embedding_d = nn.Embedding(d_vocab_size, hidden_dim, padding_idx=999)
        self.embedding_c = nn.Embedding(c_vocab_size, hidden_dim, padding_idx=0)
        self.embedding_u = nn.Embedding(u_vocab_size, hidden_dim, padding_idx=0)
        self.mix_c_d = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding_d.weight, -initrange, initrange)
        nn.init.uniform_(self.embedding_c.weight, -initrange, initrange)
        nn.init.uniform_(self.embedding_u.weight, -initrange, initrange)

    def forward(self, seqs, d, c, u):
        """

        Args:
            seqs: (batch_size, max_seq_len, representation_dim)
            d: (batch_size, max_seq_len - 1)
            c: (batch_size, max_seq_len - 1)
            u: (batch_size, max_seq_len)

        Returns:
            Sequence + AST positional encoding
        """
        d = self.embedding_d(d)  # BxL-1xd
        c = self.embedding_c(c)  # BxL-1xd
        u = self.embedding_u(u)  # BxLxd

        seqs_u = seqs + u
        d_c = torch.relu(self.mix_c_d(torch.cat((d, c), dim=2)))  # BxL-1xd
        for i in range(0, seqs_u.size(1)):
            seqs_u[:, i, :] += (d_c[:, i, :]
                                if i < d_c.shape[1] else 0) + \
                               (d_c[:, i - 1, :]
                                if i > 1 else 0)
        return seqs_u


class SyntaxPositionalEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dimension=768,
        d_vocab=100,
        c_vocab=100,
        u_vocab=100,
        d_dim=128,
        c_dim=128,
        mix_c_d_dim=128,
        padding_idx=0
    ):
        super(SyntaxPositionalEmbedding, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.d_vocab = d_vocab
        self.c_vocab = c_vocab
        self.u_vocab = u_vocab
        self.embedding_d = nn.Embedding(d_vocab, d_dim, padding_idx)
        self.embedding_c = nn.Embedding(c_vocab, c_dim, padding_idx)
        self.embedding_u = nn.Embedding(u_vocab, hidden_dimension, padding_idx)
        self.mix_c_d = nn.Linear(d_dim + c_dim, mix_c_d_dim, bias=False)
        self.mix_hidden = nn.Linear(2 * hidden_dimension, hidden_dimension, bias=False)
        self.mix_c_d_hidden = nn.Linear(mix_c_d_dim + hidden_dimension, hidden_dimension, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding_d.weight, -initrange, initrange)
        nn.init.uniform_(self.embedding_c.weight, -initrange, initrange)
        nn.init.uniform_(self.embedding_u.weight, -initrange, initrange)

    def forward(self, seqs, d, c, u):
        d = self.embedding_d(d)  # BxL-1xd
        c = self.embedding_c(c)  # BxL-1xd
        u = self.embedding_u(u)  # BxLxd

        seqs_u = seqs + u
        d_c = torch.relu(self.mix_c_d(torch.cat((d, c), dim=2)))  # BxL-1xd
        seqs_u_pairs = torch.stack([seqs_u[:, i:i + 2, :] for i in range(0, seqs_u.size(1) - 1)], dim=1)  # BxL-1x2xd
        seqs_u_pairs = seqs_u_pairs.reshape(seqs_u_pairs.shape[0], seqs_u_pairs.shape[1], -1)  # BxL-1x2d
        seqs_u_pairs = torch.relu(self.mix_hidden(seqs_u_pairs))  # BxL-1xd

        mix_c_d_hidden = torch.relu(self.mix_c_d_hidden(torch.cat((seqs_u_pairs, d_c), dim=2)))  # BxL-1xd
        for i in range(0, seqs_u.size(1)):
            seqs_u[:, i, :] += (mix_c_d_hidden[:, i, :]
                                if i < mix_c_d_hidden.shape[1] else 0) + \
                               (mix_c_d_hidden[:, i - 1, :]
                                if i > 1 else 0)
        return seqs_u
