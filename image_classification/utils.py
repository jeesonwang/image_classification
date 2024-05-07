import sys
import time
import os
import shutil
import torch

import numpy as np
from colorama import Fore
import streamlit as st
import streamlit.components.v1 as components
import ndraw

def create_save_folder(save_path, force=False, ignore_patterns=[]):
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET
              + ' already exists!', file=sys.stderr)
        if not force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                os.exit(1)
        from getpass import getuser
        tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
                                                      os.path.basename(save_path),
                                                      time.time())
        print('move existing {} to {}'.format(save_path, Fore.RED
                                              + tmp_path + Fore.RESET))
        shutil.copytree(save_path, tmp_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)

    # copy code to save folder
    if save_path.find('debug') < 0:
        shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
                                                      '*.path.tar', '*.pth',
                                                      '*.ipynb', '.*', 'data',
                                                      'save', 'save_backup',
                                                      save_path,
                                                      *ignore_patterns))


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate**2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


###############################################################
# Copied from  https://github.com/uoguelph-mlrg/Cutout
# ECL v2.0 license https://github.com/uoguelph-mlrg/Cutout/blob/master/LICENSE.md

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
###############################################################

class TrainCallback:
    def __init__(self):

        self.summary_line = None
        self.process_text = None
        self.process_bar = None
        self.loss_line = None
        self.acc_line = None
        

    def on_train_begin(self, epochs):
        self.epochs = epochs

        self.train_info = st.expander("训练信息")

        st.subheader("训练进度")
        self.process_text = st.text("0/{}".format(epochs))
        self.process_bar = st.progress(0)
        self.process_bar.progress(1 / self.epochs)
        
        st.header("训练汇总")
        self.summary_line = st.area_chart()

        st.subheader('loss曲线')
        self.loss_line = st.line_chart()

        st.subheader('accuracy曲线')
        self.acc_line = st.line_chart()


    def on_epoch_end(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.loss_line.add_rows({'train_loss': [train_loss], 'val_loss': [val_loss]})
        self.acc_line.add_rows({'train_acc': [train_acc], 'val_accuracy': [val_acc]})
        self.process_bar.progress((epoch + 1) / self.epochs)
        self.process_text.empty()
        self.process_text.text("{}/{}".format(epoch + 1, self.epochs))


    def on_batch_end(self, epoch, train_loss, train_acc):
        self.summary_line.add_rows({'loss': [train_loss], 'accuracy': [train_acc]})
    
    def on_train_info(self, info: str):
        with self.train_info:
            st.write(info)
    