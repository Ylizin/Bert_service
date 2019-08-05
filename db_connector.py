from pymongo import MongoClient

spider = MongoClient('mongodb://root:boyu42@localhost').spider
preprocess = spider.preprocess_arxiv

def read_record(batch_size = 1000):
    return preprocess.find().batch_size(1000)