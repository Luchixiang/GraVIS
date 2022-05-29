"""
Training code for C2L

"""
from __future__ import print_function

import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from loss import GraVISLoss

from model import InsResNet50
from utils import adjust_learning_rate, AverageMeter

try:
    from apex import amp, optimizers
except ImportError:
    pass


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def train_GraVIS(args, data_loader):
    train_loader = data_loader['train']
    n_data = len(train_loader)
    # model = InsResNet18(width=1)
    # model = InsResNet101(width=1)
    model = InsResNet50(width=1)

    criterion = GraVISLoss(anneal=0.2, batch_size=args.b * 2, num_id=args.b, feat_dims=128)
    # criterion = BELoss(num_class=args.b, batch_size=args.b * 20)
    criterion = criterion.cuda()

    # model_ema = torch.nn.DataParallel(model_ema)
    model = model.cuda()
    # model_ema = model_ema.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                )
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = torch.nn.DataParallel(model)
    # model = BalancedDataParallel(0, model)
    cudnn.benchmark = True

    for epoch in range(0, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        loss, prob = train(epoch, train_loader, model, criterion, optimizer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # saving the model
        if epoch % 10 == 0:
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = os.path.join(args.output,
                                     args.model + "_" + args.n + '_' + args.phase + '_' + str(
                                         args.ratio) + '_' + str(epoch) + '.pt')
            torch.save(state, save_file)
            # help release GPU memory
            del state
        torch.cuda.empty_cache()


def train(epoch, train_loader, model, criterion, optimizer):
    """
    one epoch training for instance discrimination
    """

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (input1) in enumerate(train_loader):
        # input1 shape: (bsz, 20(不同aug数量)， 3， 224， 224)
        data_time.update(time.time() - end)

        x1 = input1.view(-1, 3, 224, 224)
        bsz = x1.size(0)
        x1 = x1.float().cuda()
        # print(x1.shape, x1.device)
        # ===================forward=====================

        # ids for ShuffleBN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)
        x1 = x1[shuffle_ids]
        feat_q = model(x1)
        # print(fea)
        feat_q = feat_q[reverse_ids]
        x1 = x1[reverse_ids]
        # print(feat_q.shape)
        loss = criterion(feat_q)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        # prob_meter.update(prob.item(), bsz)

        # moment_update(model, model_ema, 0.999)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                .format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=loss_meter))
            # print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg
