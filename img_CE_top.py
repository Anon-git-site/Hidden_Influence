# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

#uncomment the appropriate dataset - cifar-10, places or INaturalist
#root = '.../cif_CEs_' 
#root = '.../plc_CEs_' 
root = '.../inat_CEs_' 


#########################################################
#cif, inat, plc
#ds='cif_'
#ds = 'plc_'
ds = 'inat_'

typ = 'reg'
runs = 3
m_eng = ['base','rem','dsm','eos']
meth = ['base_imb_','rem_','dsm_','eos_']

##########################################################


len_meth = len(meth)
n_cls = 5# 10 #uncomment 10 for cifar-10; number of classes
topk_per = 0.1

topk_meth = np.zeros((len_meth,n_cls))
topk_meth_pos = np.zeros((len_meth,n_cls))

######################################
for m in range(len_meth): 
    print(meth[m])
    n_descr = []
    n_app = []
    n_mag = []
    
    
    tp_mu_top = np.zeros((runs,n_cls))
    tp_mu_top_pos = np.zeros((runs,n_cls))
    
    
    for r in range(runs):
        f = root + meth[m] + str(r) + '.csv'
        mth = pd.read_csv(f)
        mth = mth.to_numpy()
        
        mtar = mth[:,0]
        mprd = mth[:,1]
        
        mce = mth[:,2:]
        n_feat = mce.shape[1]
        topk = int(n_feat * topk_per)
        
        for c in range(n_cls):  
            
            mctar = mtar[mtar==c]
            mcprd = mprd[mtar==c]
            mc_ce = mce[mtar==c]
            
            #TP only
            mtar_tp = mctar[mcprd==c]
            mce_tp = mc_ce[mcprd==c]
            
            mce_tp = np.mean(mce_tp,axis=0)
            mce_tp = np.squeeze(mce_tp)
            mce_tp_srt = np.sort(mce_tp)
            
            mce_sum = np.sum(mce_tp)
            
            mce_tp_srt = mce_tp_srt[::-1]
            pos_ind = np.where(mce_tp>0)
            pos_ce = mce_tp[pos_ind]
            pos_ce_sum = np.sum(pos_ce)
            topk_sum = np.sum(mce_tp_srt[:topk])
            tp_mu_top[r,c]=topk_sum / mce_sum
            tp_mu_top_pos[r,c]=topk_sum / pos_ce_sum
            
    print(tp_mu_top)
    print(tp_mu_top_pos)
    mce_mu_top = np.mean(tp_mu_top,axis=0)
    topk_meth[m,:]=mce_mu_top
    tp_mu_top_pos = np.mean(tp_mu_top_pos,axis=0)
    topk_meth_pos[m,:]=tp_mu_top_pos
    print()
           
m_eng = np.array(m_eng).reshape(-1,1)
pd1 = pd.DataFrame(data=m_eng,columns=['Meth'])
pd2 = pd.DataFrame(topk_meth)
comb = pd.concat([pd1,pd2],axis=1)
print(comb.shape)
f1='.../folder/' + ds + \
    'CE_tp_topk_' + typ + '.csv'

comb.to_csv(f1,index=False)

pd1 = pd.DataFrame(data=m_eng,columns=['Meth'])
pd2 = pd.DataFrame(topk_meth_pos)
comb = pd.concat([pd1,pd2],axis=1)
print(comb.shape)
f1='.../folder/' + ds + \
    'CE_tp_topk_pos_' + typ + '.csv'

comb.to_csv(f1,index=False)
    
   


