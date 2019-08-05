from pymongo import MongoClient
from gensim.parsing.preprocessing import *

spider = MongoClient('mongodb://root:boyu42@localhost').spider
arxiv = spider.arxiv
preprocess_arxiv = spider.preprocess_arxiv

def read_mongo():
    return arxiv.find({'sems_topics':{"$exists":True}}).batch_size(1000)

def preprocess(record):
    custom_filters = [lambda x:x.lower(),strip_punctuation,strip_multiple_whitespaces,remove_stopwords,strip_numeric,strip_short]
    fields_needed = ['_id','subjects','abstract','title','subjects','sems_topics']
    new_record = {field:record[field] for field in fields_needed}
    new_record['topic_text'] = preprocess_string(','.join(new_record['sems_topics']),custom_filters)
    new_record['title'] = preprocess_string(new_record['title'],custom_filters)
    new_record['abstract'] = preprocess_string(new_record['abstract'],custom_filters)
    return new_record

def read_cursor(cursor):
    _new_records = []
    _ids = []
    for record in cursor:
        new_record = preprocess(record)
        _new_records.append(new_record)
        _ids.append(record['_id'])
        if len(_ids) == 1000:
            arxiv.update_many({'_id':{'$in':_ids}},{'$set':{'preprocessed':True}})
            preprocess_arxiv.insert_many(_new_records)
            _ids = []
            _new_records = []

    else:
        arxiv.update_many({'_id':{'$in':_ids}},{'$set':{'preprocessed':True}})
        preprocess_arxiv.insert_many(_new_records)

    
if __name__ == '__main__':
    cursor = read_mongo()
    read_cursor(cursor)