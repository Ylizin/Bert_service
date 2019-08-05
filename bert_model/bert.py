from pytorch_transformers import *
import torch

class Inference(object):
    def __init__(self,model_dir):
        self.model = BertModel.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_id = self.tokenizer.encode(self.pad_token)
        self.max_len = 0

    def __tokenize_sentences(self,sentence1,sentence2=None):
        if sentence2:
            sentence = '{}\t{}\t{}\t{}\t{}'.format(self.cls_token,sentence1,self.sep_token,sentence2,self.sep_token)
        else:
            sentence = '{}\t{}\t{}'.format(self.cls_token,sentence1,self.sep_token)

        return self.tokenizer.encode(sentence)

    def __get_sentences_ids(self,list_of_sentences):
        '''
        if two sent embedding at one time, the list should be a list of 
        (sen1,sen2)
        '''
        if isinstance(list_of_sentences[0],tuple):
            # assert isinstance(two_sent[0],tuple),'If two_sent is True, the element should be a tuple.'
            return [self.__tokenize_sentences(tup[0],tup[1]) for tup in list_of_sentences]
        else:
            return [self.__tokenize_sentences(li) for li in list_of_sentences]
            
    def __pad_ids(self,ids):
        self.max_len = 0
        for li in ids:
            self.max_len = self.max_len if self.max_len>=len(li) else len(li)
        return torch.tensor([i+self.pad_id*(self.max_len - len(i)) for i in ids])

    def get_vec(self,list_of_str,return_pooled=False):
        assert isinstance(list_of_str,list),'get vec param must be list, wrap it please.'
        padded_ids = self.__pad_ids(self.__get_sentences_ids(list_of_str))
        embeddings,pooled_embedding = self.model(padded_ids)
        if return_pooled:
            return embeddings,pooled_embedding
        else:
            return embeddings