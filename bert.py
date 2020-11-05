import paddle
import paddle.nn as nn
from .embedding import InputEmbedding
class BERT(nn.Layer):
    def __init__(self, vocab_size: int=300, max_len: int=128,
                emb_size: int=768, n_layers: int=12, n_heads: int=8, dropout: float=0.1, pad_idx: int=0):
        super().__init__()
        self.input_emb = InputEmbedding(max_len=max_len, vocab_size=vocab_size, 
        emb_size=emb_size, pad_idx=pad_idx, dropout=dropout)
        self.transformers = nn.LayerList(
            [nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, dim_feedforward=4*emb_size, normalize_before=True)
            for _ in range(n_layers)]
        )
        self.max_len = max_len
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        self.dim_output = emb_size
        self.vocab_size = vocab_size
    def forward(self, sequence, segment):
        assert sequence.shape[1] == self.max_len, 'max_len is not euqal to length of sequence'
        mask_q = paddle.cast(sequence != self.pad_idx, dtype='float32').unsqueeze(1).unsqueeze(-1)
        
        mask_matrix = paddle.matmul(mask_q,  mask_q, transpose_y=True)
        mask_matrix = paddle.tile(mask_matrix, repeat_times=[1, self.n_heads, 1, 1])
        
        x = self.input_emb(sequence, segment)
        # print('x.shape is :', x.shape)
        # print(mask_matrix.shape)
        for transformer in self.transformers:
            x = transformer(x, mask_matrix)
        return x

class BERTLM(nn.Layer):
    def __init__(self, bert: BERT):
        super().__init__()

        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.dim_output)
        self.mask_lm  = MaskedLanguageModel(self.bert.dim_output, self.bert.vocab_size)
    def forward(self, sentence, segment, pretrain_mode=True):
        x = self.bert(sentence, segment)
        if pretrain_mode is not True:
            return x
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Layer):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(axis=-1)
    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(nn.Layer):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(axis=-1)
    def forward(self, x):
        return self.softmax(self.linear(x))