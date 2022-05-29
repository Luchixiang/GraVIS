from torch.utils.data import DataLoader

from dataset import *
from utils import *
from torchvision import transforms, datasets
import torch
import torchio.transforms
import copy


def GraVIS_isic_pretask(args):
    print('using GraVIS pretrain on isic')
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    train_transform.transforms.append(Cutout(n_holes=3, length=32))
    train_file = './isic_seg_train.txt'
    train_imgs = get_monu_list(args, train_file)
    train_imgs = train_imgs[:int(len(train_imgs) * args.ratio)]
    train_dataset = GraVISDataset(train_imgs, transform=train_transform, two_crop=True)
    print(len(train_dataset))
    train_sampler = None
    dataloader = {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.b, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    dataloader['train'] = train_loader
    dataloader['eval'] = train_loader
    return dataloader
