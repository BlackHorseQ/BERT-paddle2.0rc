import paddle.nn as nn
import paddle
import numpy as np

class PositionalEmbedding(nn.Layer):
    def __init__(self, posistion: int=60, d_model: int=30):
        super().__init__()
        
        pos_enc = paddle.zeros(shape=[posistion, d_model], dtype='float32') 
        pos     = paddle.arange(start=0, end=posistion, dtype='float32').unsqueeze(1)
        dim     = paddle.arange(start=0, end=d_model, step=2, dtype='float32')
        div_den = paddle.pow(paddle.to_tensor(np.array([10000]), dtype='float32'), -(dim/d_model))
        pos_enc[:, 0::2] = paddle.sin(pos * div_den)
        pos_enc[:, 1::2] = paddle.cos(pos * div_den)
        pos_enc.stop_gradient = True
        self.register_buffer('pos_enc', pos_enc)

    def forward(self):
        return self.pos_enc.unsqueeze(0)
class InputEmbedding(nn.Layer):

    def __init__(self, max_len: int=20, vocab_size: int=3000, emb_size: int=128, pad_idx: int=0, dropout: float=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.segment_embedding = nn.Embedding(3, emb_size)
        self.position_embedding = PositionalEmbedding(posistion=max_len, d_model=emb_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment):
        x = self.token_embedding(sequence) + self.position_embedding() + self.segment_embedding(segment)

        return self.dropout(x)