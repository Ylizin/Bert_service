from torch.utils.data import Dataset
import numpy as np
from .config import configs
from .db_connector import read_record
from functools import partial
from torch.utils.data import DataLoader

class MongoData(Dataset):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.corpus = None
        self.word2id = {}
        self.id2word = {}   
        self.word_cnt = 0
        self.title_array = []
        self.abstract_array=[]
        self.topic_array = []
        self.__read_from_mongo()

    def __read_from_mongo(self):
        cursor = read_record()
        for record in cursor:
            self.title_array.append(self.__vectorize_list(record['title'],100))
            self.abstract_array.append(self.__vectorize_list(record['abstract'],500))
            self.topic_array.append(self.__vectorize_list(record['topic_text'],100))

    def __get_word_id(self,word):
        '''
            invoke this function will (add if not in)/get an index of the word in the corpus
        '''
        if word in self.word2id:
            return self.word2id[word]
        else:
            # 0 is for OOV
            self.word_cnt += 1
            self.word2id[word] = self.word_cnt
            self.id2word[self.word_cnt] = word
            return self.word_cnt
        
    def __vectorize_list(self,words_list,length):
        '''
            words_list: the list to be vectorized
            length: the length that the np array will be, if the list is longer than the length,
            the word list will be cutted
        '''
        words_list = words_list[:length]
        _idx_array = np.array(list(map(self.__get_word_id,words_list)))
        _idx_padding = np.full(length-len(words_list),0)
        return np.append(_idx_array,_idx_padding)
        
    def __getitem__(self, index):
        return self.title_array[index],self.abstract_array[index],self.topic_array[index]
    
    def __len__(self):
        return len(self.title_array)


if __name__ == '__main__':
    dataset = MongoData('')
    dataloader = DataLoader(dataset,batch_size = 10)
    for i in dataloader:
        print(i[0].shape)
        break
    
    