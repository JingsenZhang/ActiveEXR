import math
import torch
import random
from utils import sentence_format


class Batchify:
    def __init__(self, data, word2idx, text_avaiable_num, seq_len=15, batch_size=128, shuffle=False):
        self.sample_num = len(data)
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        u, i, r, t, f, s = [], [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append(sentence_format(x['text'], seq_len, pad, bos, eos))
            f.append([x['feature']])
            s.append(x['sco'])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.prob = torch.ones_like(self.user, dtype=torch.float).contiguous()
        self.text_label = torch.cat([torch.ones(text_avaiable_num, dtype=torch.float),
                                     torch.zeros(self.sample_num - text_avaiable_num, dtype=torch.float)]).contiguous()
        self.batch_size = batch_size
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.shuffle = shuffle
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        feature = self.feature[index]  # (batch_size, 1)
        prob = self.prob[index]
        text_label = self.text_label[index]
        return user, item, rating, seq, feature, prob, text_label

    def get_part_data(self, index):
        data_dict = {}
        data_dict['user'] = self.user[index]  # (batch_size,)
        data_dict['item'] = self.item[index]
        data_dict['rating'] = self.rating[index]
        data_dict['seq'] = self.seq[index]  # (batch_size, seq_len)
        data_dict['feature'] = self.feature[index]  # (batch_size, 1)
        return data_dict


class Batchify_generator:
    def __init__(self, u, i, batch_size=128, shuffle=False):
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.seq = torch.zeros((len(u), 1), dtype=torch.int64).contiguous()  # inputs: <bos>
        self.batch_size = batch_size
        self.sample_num = len(u)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.shuffle = shuffle
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        seq = self.seq[index]
        return user, item, seq
