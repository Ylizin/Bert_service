from multiprocessing.connection import Listener
from bert import Inference
import os

socket_file = './BertSocket'
bert_dir = './model'
def __listen():
    server = None
    try:
        server = Listener(socket_file,authkey=b'boyu42')
    except Exception as e:
        os.unlink(socket_file)
        server = Listener(socket_file,authkey=b'boyu42')
    return server

def __init_Bert():
    return Inference(bert_dir)

def init_server():
    server = __listen()
    model = __init_Bert()
    conn = None
    while True:
        try:
            if not conn:
                conn = server.accept()
            list_of_str = conn.recv()
            if not isinstance(list_of_str,list):
                conn.send(-1)
            elif not (isinstance(list_of_str[0],str) or isinstance(list_of_str[0],tuple)):
                conn.send(-1)
            tensors = model.get_vec(list_of_str).cpu().detach().numpy() #这里一定需要使用numpy，tensor是无法这样发送的
            # conn.send(list_of_str)
            conn.send(tensors)
        except Exception as e:
            server = __listen()

if __name__ =='__main__':
    init_server()    
