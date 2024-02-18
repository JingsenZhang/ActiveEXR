import os
import heapq
import pickle


class WordDictionary:  # word & feature
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']  # list
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}  # dict:{'<bos>':0, '<eos>':1, '<pad>':2, '<unk>':3}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):  # add word & record the word count
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        # print('word_size: ', len(self.__word2count))
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        # print('id2: ', self.idx2word[:10])


class EntityDictionary:  # user & item
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:

    def __init__(self, data_path, index_dir, vocab_size):
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set()
        self.train, self.valid, self.test, self.unlabel, \
        self.train_index, self.valid_index, self.test_index, self.unlabel_index = self.load_data(data_path, index_dir)
        self.train_size = len(self.train)
        self.valid_size = len(self.valid)
        self.test_size = len(self.test)
        self.unlabel_size = len(self.unlabel)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            (fea, adj, tem, sco) = review['template']
            self.word_dict.add_sentence(tem)
            self.word_dict.add_word(fea)  # add the feature into corpus
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            fea_idx = self.word_dict.word2idx.get(fea, self.__unk)
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': self.seq2ids(tem),
                         'feature': fea_idx,
                         'sco': 5 if sco == 1 else sco})
            self.feature_set.add(fea_idx)  # feature ID (token id)
        train_index, valid_index, test_index, unlabel_index = self.load_index(index_dir)
        train, valid, test, unlabel = [], [], [], []
        for idx in train_index:
            train.append(data[idx])
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        for idx in unlabel_index:
            unlabel.append(data[idx])

        return train, valid, test, unlabel, train_index, valid_index, test_index, unlabel_index

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'valid.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'unlabel.index'), 'r') as f:
            unlabel_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index, unlabel_index

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]
