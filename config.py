import argparse
import torch

configs = argparse.ArgumentParser('Global Configs')
configs.add_argument('-cuda',type = bool,default=torch.cuda.is_available())


configs = configs.parse_args()