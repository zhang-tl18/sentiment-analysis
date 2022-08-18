import torch
from torch.utils.data import Dataset
import numpy as np
from gensim.models import KeyedVectors

class My_Dataset(Dataset):
    def __init__(self, file_path, word2vec_path='./Dataset/wiki_word2vec_50.bin', embedding_size=50, max_text_len=679) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.max_text_len = max_text_len
        self.data = []
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label, words = line.split('\t')
                label = int(label)
                words = words.split() #['word', ... , 'word']
                self.data.append([words, label])
    
    def __getitem__(self, index):
        words = self.data[index][0]
        vecs = []
        for x in words:
            if x in self.word2vec:
                vecs.append(self.word2vec[x])
            else:
                vecs.append([0]*self.embedding_size)
        data = torch.cat([torch.FloatTensor(np.array(vecs)), torch.zeros(self.max_text_len-len(vecs), self.embedding_size)])
        label = torch.tensor(self.data[index][1])
        return data, label
    
    def __len__(self):
        return len(self.data)