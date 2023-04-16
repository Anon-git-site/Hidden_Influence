# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from numpy import linalg as LA
import resnet_cifar_FE as models

from utils import *
from imbalance_cifar import IMBALANCECIFAR10

##########################################################################

t00 = time.time()
t0 = time.time()

torch.set_printoptions(precision=4, threshold=20000, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

print('model names ', model_names)

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', 
                    help='dataset setting')

parser.add_argument('-a', '--arch', metavar='ARCH',
                    default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet32)')

parser.add_argument('--loss_type',
                    default="CE",
                    type=str, help='loss type')



parser.add_argument('--imb_type', default="exp",
                    type=str, help='imbalance type')

parser.add_argument('--imb_factor', default=0.01,
                    type=float, help='imbalance factor')

parser.add_argument('--train_rule',
                    default='None',
                    type=str,
                    help='data sampling strategy for train loader')

parser.add_argument('--n_cls', 
                    default=10, 
                    type=int,
                    help='number of classes')

parser.add_argument('--n_feat', 
                    default=64, 
                    type=int,
                    help='number of classes')

parser.add_argument('--rand_number', 
                    default=0, 
                    type=int,
                    help='fix random number for data sampling')

parser.add_argument('--exp_str', default='0', type=str,
                    help='number to indicate which experiment it is')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs',
                    default=1,
                    type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size',
                    default=1, 
                    type=int,
                    metavar='N',
                    help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    help='GPU id to use.')

parser.add_argument('--data_path',
        default='.../data/', 
        type=str,
        help='data path.')


parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed(0)

###################################################################
best_acc1 = 0
best_acc = 0  # best test accuracy

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))
print()

args.store_name = '_'.join([args.dataset, args.arch, args.loss_type,
                            args.train_rule, args.imb_type, 
                            str(args.imb_factor), args.exp_str])

#iteratively uncomment the respective models to generate CE files
#for CIFAR-10, which can be found in the models/cifar folder
"""
#base
models_ = [
".../base_mod_133_0.pth",   
".../base_mod_186_10.pth",
".../base_mod_186_100.pth",   
    ]

#rem
meth = 'rem_'
models_ = [
".../rem_mod_115_0.pth",   
".../rem_mod_162_10.pth",
".../rem_mod_146_100.pth",   
    ]


meth = 'eos_'
#eos
models_ = [
".../EOS_0_best.pth",   
".../EOS_10_best.pth",
".../EOS_100_best.pth",   
    ]
"""
#dsm
meth = 'dsm_'
models_ = [
".../DSM_0_best.pth",   
".../DSM_10_best.pth",
".../DSM_100_best.pth",   
    ]


n_descr = []
n_app = []
n_hist = []
max_ns = []

for m in range(len(models_)):

    model = models.__dict__[args.arch](num_classes=args.n_cls, 
                                       use_norm=False)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    model.load_state_dict(torch.load(models_[m]))

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    val_dataset = datasets.CIFAR10(root=args.data_path,
                               train=False,
                               download=True, transform=transform_val)
    
    n_idx = len(val_dataset.targets) 
    idx = np.arange(n_idx)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,#100, 
        shuffle=False,
        num_workers=args.workers, pin_memory=True)  

    wt = model.linear.weight
    wt = wt.detach().cpu().numpy()
    print(wt.shape) #(10, 64)

    bias = model.linear.bias
    bias = bias.detach().cpu().numpy()
    print(bias.shape)#,bias) #(10, 64)

    print('wt norm',LA.norm(wt))
    print('bias norm',LA.norm(bias))

    ############################################
    
    model.eval()

    all_FE = []   
    all_preds = []
    all_targets = []
    nxt_pred = []

    for i, (input, target) in enumerate(val_loader):
    
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
    
        output, out1 = model(input)

        out1 = out1.detach().cpu().numpy()
        
        m1 = nn.Softmax(dim=1)
        soft = m1(output)
        values, pred = torch.max(soft, 1)
    
        preds = torch.argsort(output, dim=1)
        preds = preds.cpu().numpy()
        preds = np.squeeze(preds)
        preds = preds[::-1]
    
        nxt_pred.append(preds[1])

        all_FE.extend(out1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    all_FE1 = np.array(all_FE)
    
    all_tar = np.array(all_targets)
    
    all_p = np.array(all_preds)
    
    all_nxt = np.array(nxt_pred)
    
    args.n_CE = all_FE1.shape[1]

    all_CE = np.zeros((all_FE1.shape[0],all_FE1.shape[1]))
    nxt_logit = np.zeros(all_FE1.shape[0])

    for i in range(len(all_preds)):    
        
        p = all_p[i]
        
        fe = all_FE1[i,:]
        CE = fe * wt[p,:]
        
        all_CE[i,:]=CE
    
        p = all_nxt[i]
        
        CE = fe * wt[p,:]
        
        nxt_logit[i]=np.sum(CE) + bias[p]
    
    
    #################################################################
    ce_idx = []
    ce_tar = []
    ce_mags = []
    
    ce_lists = []
    ce_extend = []
    ce_len = []
    ce_len2 = []
    ce_p = []

    n_tps = np.zeros(args.n_cls)

    for c in range(args.n_cls):
        
        ce_c = all_CE[all_tar==c]
        t_c = all_tar[all_tar==c]
        p_c =  all_p[all_tar==c]
        l_c = nxt_logit[all_tar==c]
        i_c = idx[all_tar==c]
        
        #all
        ce_tp = ce_c
        lc_tp = l_c
        t_tp = t_c
        i_tp = i_c
        
        
        n_tps[c]=len(ce_tp)
        
        all_mag = []
    
        all_ces = []
        all_ces_ext = []
        all_ces_cnt = []
        all_ces_cnt2 = []
        
        
        for i in range(len(ce_tp)):
        
            ce = ce_tp[i,:]
            ce_srt = np.sort(ce)
            ce_srt = ce_srt[::-1]
        
            ce_arg = np.argsort(ce)
            ce_arg = ce_arg[::-1]
            
            neg_ind = np.where(ce_srt<0)
            ce_neg = ce_srt[neg_ind]
            
            neg_sum = np.sum(ce_neg)
            
            ces = []
            cmag = []
            
            ce_sum=0
            count = 0
            #in some cases bias exceeds next logit so no enters
            while (ce_sum + bias[c] + neg_sum) < lc_tp[i]:
                if count > (args.n_feat-1):
                
                    break
                ces.append(ce_arg[count])
                cmag.append(ce_srt[count])
                ce_sum+=ce_srt[count]
                count+=1
            all_ces.append(ces)
            all_ces_ext.extend(ces)
            all_ces_cnt.append(len(ces))
            all_ces_cnt2.append(len(ces))
            all_mag.append(cmag)
        
        ce_lists.append(all_ces)
        ce_extend.append(all_ces_ext)
        ce_len.append(all_ces_cnt)
        ce_len2.extend(all_ces_cnt)
        ce_idx.extend(i_tp)
        ce_tar.extend(t_tp)
        ce_mags.append(all_mag)
        ce_p.extend(p_c)
    
    ce_len2 = np.array(ce_len2)
    max_clen = np.max(ce_len2)
    
    #######################################################
    from collections import Counter

    n_to_describe = np.zeros(args.n_cls)

    topk = []

    for c in range(len(ce_extend)):
    
        cext = ce_extend[c]
    
        cdict = Counter(cext)
        
        ind, val = list(cdict.keys()),list(cdict.values())
    
        inda = np.argsort(val)
        vala = np.sort(val)
        inda = inda[::-1]
        vala = vala[::-1]
        vala = np.array(vala)
        
        n_appear = vala/n_tps[c]
    
        topk.append(n_appear)
        n_to_describe[c] = len(vala)
       

    max_n = int(np.max(n_to_describe))
    max_ns.append(max_n)
    max_n
    
    idxs = ['I' + str(z) for z in range(max_clen)]
    mags = ['M' + str(z) for z in range(max_clen)]
    
    len_ce = len(ce_tar)
    
    idx_g = np.zeros((len_ce,max_clen))
    mag_g = np.zeros((len_ce,max_clen))
    
    c_lens = np.zeros(len_ce)
    
    total=0
    for q in range(len(ce_lists)):
        cel = ce_lists[q]
        cem = ce_mags[q]
        count=0
        for r in range(len(cel)):
            inds = cel[r]
            inds = np.array(inds)
            
            ms = cem[r]
            ms = np.array(ms)
            
            num = len(ms)
            
            idx_g[total,:num]=inds
            mag_g[total,:num]=ms
            c_lens[total]=num
            count+=1
            total+=1
    
    print(c_lens)
    print('total',total)
    ce_tar = np.array(ce_tar).reshape(-1,1)
    pd1 = pd.DataFrame(data=ce_tar,columns=['Tar'])
    pd2 = pd.DataFrame(data=ce_idx,columns=['IDX'])
    pd3 = pd.DataFrame(data=idx_g,columns=idxs)
    pd4 = pd.DataFrame(data=mag_g,columns=mags)
    pd5 = pd.DataFrame(data=c_lens,columns=['Len'])
    pd6 = pd.DataFrame(data=ce_p,columns=['Pred'])
    
    comb = pd.concat([pd1,pd6,pd2,pd5,pd3,pd4],axis=1)
    print(comb.shape)
    
    f1='.../cif_CE_all_' \
        + meth + str(m) + '.csv' 
    comb.to_csv(f1,index=False)






