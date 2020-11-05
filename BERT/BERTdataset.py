from paddle.io import Dataset
import pandas as pd
import tqdm
import random
class BERTDataset(Dataset):

    def __init__(self, corpus_path:str, vocab:dict, seq_len:int=128): 
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.lines = pd.read_csv(corpus_path, sep='\t')
        self.corpus_lines = self.lines.shape[0]
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        t1, t2, is_next_label = self.get_sentence(idx)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab['[CLS]']] + t1_random + [self.vocab['[SEP]']]
        t2 = t2_random + [self.vocab['[SEP]']]

        t1_label = [self.vocab['[PAD]']] + t1_label + [self.vocab['[PAD]']]
        t2_label = t2_label + [self.vocab['[PAD]']]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"sentence": bert_input,
                  "bert_label": bert_label,
                  "segment": segment_label,
                  "is_next": is_next_label}
        return output

    def random_word(self, sentence):
        tokens = ' '.join(sentence)
        tokens = list(tokens)
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% 随机mask
                if prob < 0.8:
                    tokens[i] = self.vocab['[MASK]']

                # 10% 随机选取
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% 不替换
                else:
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])

                output_label.append(self.vocab.get(token, self.vocab['[UNK]']))

            else:
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(0)
        return tokens, output_label

    def get_sentence(self, idx):
        t1, t2, _ = self.lines.iloc[idx].values

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.lines.iloc[random.randrange(self.lines.shape[0])].values[1], 0
