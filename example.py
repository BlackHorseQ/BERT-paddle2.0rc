import pandas as pd
import paddle
import paddle.nn as nn
import numpy as np
from BETT.BERTdataset import BERTDataset
from BETT.bert import BERT, BERTLM
from BETT.trainModle import Model
if __name__ == '__main__':
​
    ## 预训练， 加载词汇表
    vocab = pd.read_csv('./train/vocab.txt', sep='\t', names=['key', 'value'])
    vocab = vocab.set_index('key')
    vocab = vocab.to_dict()['value']
​
    ## BERT单元
    bert = BERT(max_len=30, vocab_size=len(vocab))
    bertlm = BERTLM(bert)
​
    
    opt = paddle.optimizer.Adam(parameters=bertlm.parameters())
    model = Model(bertlm, optimizer=opt, log_freq=10)
    ## 数据格式为tsv， text_a, text_b, label, 详见train/*.tsv
    train_data = BERTDataset(corpus_path='train/train.tsv', vocab=vocab, seq_len=30)
    test_data = BERTDataset(corpus_path='train/test.tsv', vocab=vocab, seq_len=30)
    ##开始预训练
    model.fit(train_data, test_data, epoch=10, batch_size=32, save_epochs=9)
​
    #保存预训练结果
    paddle.fluid.save_dygraph(model.model.state_dict(), file_path)
    paddle.fluid.save_dygraph(model.opt.state_dict(), file_path)
    
    
    ##使用预训练模型
    bert = BERT(max_len=30, vocab_size=len(vocab))
    bertlm = BERTLM(bert)
    file_path = 'bert'
    opt = paddle.optimizer.Adam(parameters=bertlm.parameters())
    para_state_dict, opt_state_dict = paddle.fluid.load_dygraph(file_path)
    bertlm.set_state_dict(para_state_dict)
    opt.set_state_dict(opt_state_dict)
​
    ##使用时需要调用 eval()
    bertlm.eval()
    #以QA问题举例，输入 sentence, segement shape为[batch_size, sentence_len]的tensor
    #sentence为句子的index，segement为句子的分割向量由1和2组成的序列
    sentence = paddle.to_tensor(np.array([[i for i in range(30)]]), dtype='int64')
    segment  = paddle.to_tensor(np.array([[1]*14 + [2]*16]))
    ##输出特征shape为[batch_size, length, dim]的特征
    output = bertlm(sentence, segment, pretrain_mode=False)