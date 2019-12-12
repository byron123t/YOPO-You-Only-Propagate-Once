from torchsummary import summary
from config import config
from dataset import load_test_dataset
from network import create_network
from network import test

from utils.misc import load_checkpoint

import argparse
import torch
import numpy as np
import os
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
from tqdm import tqdm
from typing import Tuple, List, Dict
import pickle


def my_torch_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    '''
    param output, target: should be torch Variable
    '''
    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


def my_eval_one_epoch(net, batch_generator,  DEVICE=torch.device('cuda:0'), AttackMethod = None):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()
    correct_indices = None
    natural_indices = None

    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            pred = net(data)
            predictions = np.argmax(pred.cpu().numpy(), axis=1)
            correct_labels = label.cpu().numpy()
            if natural_indices is None:
                natural_indices = np.where(predictions==correct_labels)[0]
            else:
                natural_indices = np.append(natural_indices, np.where(predictions==correct_labels)[0])
            
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, data, label)

            with torch.no_grad():
                pred = net(adv_inp)
                predictions = np.argmax(pred.cpu().numpy(), axis=1)
                correct_labels = label.cpu().numpy()
                if correct_indices is None:
                    correct_indices = np.where(predictions==correct_labels)[0]
                else:
                    correct_indices = np.append(correct_indices, np.where(predictions==correct_labels)[0])

                acc = my_torch_accuracy(pred, label, (1,))
                adv_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(adv_accuracy.mean)

        pbar.set_postfix(pbar_dic)

        adv_acc = adv_accuracy.mean if AttackMethod is not None else 0

    print('Natural Samples', natural_indices.shape)
    print('Adversarial Samples', correct_indices.shape)

    return clean_accuracy.mean, adv_acc


def process_single_epoch():
    print('**************')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
    args = parser.parse_args()


    DEVICE = torch.device('cuda:{}'.format(args.d))
    torch.backends.cudnn.benchmark = True

    net = create_network()
    net.to(DEVICE)

    nat_val = load_test_dataset(10000, natural=True)
    adv_val = load_test_dataset(10000, natural=False)

    AttackMethod = config.create_evaluation_attack_method(DEVICE)

    filename = '../ckpts/4leaf-epoch32.checkpoint'
    print(filename)
    if os.path.isfile(filename):
        load_checkpoint(filename, net)

    print('Evaluating Natural Samples')
    clean_acc, adv_acc = my_eval_one_epoch(net, nat_val, DEVICE, AttackMethod)
    print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))

    print('Evaluating Adversarial Samples')
    clean_acc, adv_acc = my_eval_one_epoch(net, adv_val, DEVICE, AttackMethod)
    print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))


if __name__ == '__main__':
    process_single_epoch()
