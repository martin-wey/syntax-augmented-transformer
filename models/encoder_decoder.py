import math

import torch
import torch.nn as nn
from torch import Tensor

from .syntax_positional_embedding import SimpleSyntaxPositionalEncoding


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    if src is None:
        src_mask = None
    else:
        src_seq_len = src.shape[1]
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool).to(src.device)

    if tgt is None:
        tgt_mask = None
    else:
        tgt_seq_len = tgt.shape[1]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

    return src_mask, tgt_mask


class HighwayGate(nn.Module):
    def __init__(self, in_out_dim, bias=True):
        super(HighwayGate, self).__init__()
        self.gateway_linear = nn.Linear(in_out_dim, in_out_dim, bias=bias)

    def forward(self, x1, x2):
        coef = torch.sigmoid(self.gateway_linear(x1))
        return coef * x1 + (1 - coef) * x2


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_index: int = 0,
    ):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_index)
        self.positional_embedding = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.d_model = hidden_dim

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)

    def get_embeddings(self, inputs):
        tgt_embds = self.embedding(inputs) * math.sqrt(self.d_model)
        tgt_embds = self.positional_embedding(tgt_embds)
        return tgt_embds.permute(1, 0, 2)

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_embedding(src)
        # permute batch_size and seq_len
        src = src.permute(1, 0, 2)
        output = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerEncoderSyntax(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        syntax_gate: bool,
        pad_index: int = 0,
    ):
        super(TransformerEncoderSyntax, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_index)
        self.positional_embedding = PositionalEncoding(hidden_dim)
        self.syntax_encoding = SimpleSyntaxPositionalEncoding()
        if syntax_gate:
            self.syntax_gate = HighwayGate(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.d_model = hidden_dim

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)

    def get_embeddings(self, inputs):
        tgt_embds = self.embedding(inputs) * math.sqrt(self.d_model)
        tgt_embds = self.positional_embedding(tgt_embds)
        return tgt_embds.permute(1, 0, 2)

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor, d: Tensor, c: Tensor, u: Tensor) -> Tensor:
        src_embds = self.embedding(src) * math.sqrt(self.d_model)
        src_positional = self.positional_embedding(src_embds)
        src = self.syntax_encoding(src_positional, d, c, u)
        if hasattr(self, 'syntax_gate'):
            src = self.syntax_gate(src_positional, src)

        # permute batch_size and seq_len
        src = src.permute(1, 0, 2)
        output = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float
    ):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.d_model = hidden_dim

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        return self.transformer_decoder(tgt=tgt,
                                        memory=memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)


class TransformerEncoderDecoder(nn.Module):
    # Code adapted from: https://pytorch.org/tutorials/beginner/translation_transformer.html
    def __init__(
        self,
        encoder: nn.TransformerEncoder,
        decoder: nn.TransformerDecoder,
        d_model: int,
        vocab_size: int
    ):
        super(TransformerEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dense = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.act = nn.LogSoftmax(dim=-1)

        self.init_weights()
        self.tie_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.lm_head.weight, -initrange, initrange)

    def tie_weights(self):
        """Optionally tie weights as in:
            "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            https://arxiv.org/abs/1608.05859
            and
            "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            https://arxiv.org/abs/1611.01462
        """
        self.lm_head.weight = self.encoder.embedding.weight

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, d=None, c=None, u=None):
        src_mask, tgt_mask = create_mask(src, tgt)
        if d is not None and c is not None and u is not None:
            encoder_output = self.encoder(src, src_mask, src_padding_mask, d, c, u)
        else:
            encoder_output = self.encoder(src, src_mask, src_padding_mask)

        tgt_embds = self.encoder.embedding(tgt) * math.sqrt(self.encoder.d_model)
        tgt_embds = self.encoder.positional_embedding(tgt_embds)
        tgt_embds = tgt_embds.permute(1, 0, 2)

        output = self.decoder(tgt=tgt_embds,
                              memory=encoder_output,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        h = torch.relu(self.dense(output)).permute(1, 0, 2)
        logits = self.lm_head(h)
        return logits

    def predict(self, src, context, context_mask):
        _, tgt_mask = create_mask(None, src)
        tgt_embds = self.encoder.embedding(src) * math.sqrt(self.encoder.d_model)
        tgt_embds = self.encoder.positional_embedding(tgt_embds)
        # permute batch_size and seq_len
        tgt_embds = tgt_embds.permute(1, 0, 2)

        output = self.decoder(tgt=tgt_embds,
                              memory=context,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=None,
                              memory_key_padding_mask=context_mask)
        h = torch.relu(self.dense(output)).permute(1, 0, 2)[:, -1, :]
        out = self.act(self.lm_head(h)).data
        return out


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Beam(object):
    """Source: https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py"""
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = torch.div(bestScoresId, numWords, rounding_mode='floor')
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
