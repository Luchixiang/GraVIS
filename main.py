import argparse
import os
import warnings

import torch.backends.cudnn
import sys
from data import GraVIS_isic_pretask
from train import train_GraVIS

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/home/luchixiang/isic/ISIC-2017_Training_Data/0',
                        help='path to dataset')
    parser.add_argument('--b', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output', default='./model_genesis_pretrain', type=str, help='output path')
    parser.add_argument('--workers', default=4, type=int, help='num of workers')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='gpu indexs')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--ratio', default=1.0, type=float, help='ratio of data used for pretraining')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True
    data_loader = GraVIS_isic_pretask(args)
    train_generator = data_loader['train']
    valid_generator = data_loader['eval']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Total CUDA devices: ", torch.cuda.device_count())
    intial_epoch = 0
    sys.stdout.flush()
    train_GraVIS(args, data_loader)
