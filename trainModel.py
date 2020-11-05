from paddle.io import Dataset
import paddle
import paddle.nn as nn
from paddle import fluid
from paddle.optimizer import Optimizer
import numpy as np
from tqdm import tqdm
from .bert import BERTLM

class Model():
    def __init__(
        self, bert: BERTLM,optimizer: Optimizer, 
        with_cuda: bool=True, log_freq: int=10
    ):      
        assert paddle.is_compiled_with_cuda, 'the version of padddle is not compiled with the cuda'
        assert with_cuda is True, 'warning: do you want to train bert using cpu? are you kidding me?'

        self.model = bert    
        self.opt = optimizer

        self.criterion_sentence = nn.NLLLoss()
        self.criterion_tokens   = nn.NLLLoss(ignore_index=0)
        self.log_freq   = log_freq
        list_n_elemnets = []
        for p in self.model.parameters():
            n = 1
            for i in p.shape:
                n *= i
            list_n_elemnets.append(n)
        print('Total Parameters:', sum(list_n_elemnets))
    def get_reader(self, dataset: Dataset, batch_size: int=32):

        def reader():
            sentence = []
            segment = []
            bert_label = []
            is_next = []
            for i, data in enumerate(dataset):
                sentence.append(data['sentence'])
                segment.append(data['segment'])
                bert_label.append(data['bert_label'])
                is_next.append(data['is_next'])
                if i % batch_size == 0 and i != 0:
                    yield np.array(sentence), np.array(segment), np.array(bert_label), np.array(is_next)
                    sentence = []
                    segment = []
                    bert_label = []
                    is_next = []
            if len(segment)>0:
                yield np.array(sentence), np.array(segment), np.array(bert_label), np.array(is_next)
                sentence = []
                segment = []
                bert_label = []
                is_next = []
        return reader
    def fit(self, train_data: Dataset, test_data: Dataset=None,
        batch_size:int=32, epoch: int=10, save_epochs: int=1, file_path:str="bert"):
        for i in range(epoch):
            self.iteration(epoch=i, data_loader=train_data, train=True, batch_size=batch_size)
            if test_data is not None:
                self.iteration(epoch=i, data_loader=test_data, train=False, batch_size=batch_size)
            if i % save_epochs:
                self.save(epoch=i,  file_path)
        return
    def predict(self, test_data: Dataset):
        return self.iteration(epoch=0, data_loader=test_data, train=False)

    def iteration(self, epoch, data_loader: Dataset, train=True, batch_size: int=32):
        mode = 'train' if train else 'test'
        data_reader = self.get_reader(data_loader, batch_size)
        avg_loss = 0.0
        total_sentence_correct = 0
        total_element = 0
        if train:
            self.model.train()
        else:
            self.model.eval()

        for i, data in enumerate(data_reader()):
            sentence, segment, bert_label, is_next = data
            
            sentence = paddle.to_tensor(sentence)
            segment = paddle.to_tensor(segment)
            is_next = paddle.to_tensor(is_next)
            bert_label = paddle.to_tensor(bert_label)
            

            next_sentence_output, mask_lm_output = self.model.forward(sentence, segment)
            next_loss = self.criterion_sentence(next_sentence_output, is_next)

            mask_loss = self.criterion_tokens(paddle.transpose(mask_lm_output, perm=[0, 2, 1]), bert_label)
            loss = mask_loss + next_loss

            if train:
                loss.backward()
                self.opt.step()
                self.opt.clear_grad()
            list_sentence_correct = paddle.cast(next_sentence_output.argmax(axis=-1).equal(is_next), dtype='int64')
            sentence_correct = list_sentence_correct.sum().numpy()

            avg_loss += loss.numpy()

            total_sentence_correct += sentence_correct
            total_element += is_next.shape[0]

            post_fix = {
                'mode': mode,
                "epoch": epoch,
                'iter': i,
                'avg_loss': avg_loss/(i+1),
                'avg_acc': total_sentence_correct/ total_element * 100,
                'loss': loss.numpy()
            }
            if i% self.log_freq == 0:
                print(str(post_fix))
        return 
    def save(self, epoch:int=0, file_path:str="bert"):
        fluid.save_dygraph(self.model.state_dict(), file_path+f'_epochs_{epoch}')
        fluid.save_dygraph(self.opt.state_dict(), file_path+f'_epochs_{epoch}')
        return

    def load(self, file_path:str="output/bert"):
        para_state_dict, opt_state_dict = fluid.load_dygraph(file_path)
        self.model.set_state_dict(para_state_dict)
        self.opt.set_state_dict(opt_state_dict)
        return
         
        