# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

#uncomment for LG model CE data - ozone, scene, coil, thyroid, us crime
#root = '.../1_CE_tp_fp/oz_CEs_' 
#root = '.../1_CE_tp_fp/sc_CEs_' 
#root = '.../1_CE_tp_fp/co_CEs_' 
#root = '.../1_CE_tp_fp/th_CEs_' 
#root = '.../1_CE_tp_fp/us_CEs_' 

#uncomment appropriate SVM CE data - ozone, scene, coil, thyroid, us crime
root = '.../1_CE_tp_fp/oz_SV_CEs_' 
#root = '.../1_CE_tp_fp/sc_SV_CEs_' 
#root = '.../1_CE_tp_fp/co_SV_CEs_' 
#root = '.../1_CE_tp_fp/th_SV_CEs_' 
#root = '.../1_CE_tp_fp/us_SV_CEs_' 


#########################################################

ds_eng = ['oz','sc','co','th','us']
runs = 5
typ = 'SV' #'LG' #uncomment depending on whether SVM or LG model

ds =  'oz_' #the specific dataset abbreviation
m_eng = ['base','SM','AD','rem'] #base imbalanced, SMOTE, ADASYN, REMIX
meth = ['base_','SM_','AD_','rem_']

##########################################################

len_meth = len(meth)
n_cls = 2
topk_per = 0.1

#express as pos even tho may have negative sign because positively
#contribute to the prediction
#topk_meth = np.zeros((len_meth,n_cls))
topk_meth_pos = np.zeros((len_meth,n_cls))

######################################
for m in range(len_meth): 
    print(meth[m])
    
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
            
            #then prediction based on neg signed CE
            #neg are already sorted at front
            if c == 0:
                pos_ind = np.where(mce_tp<0)
                pos_ce = mce_tp[pos_ind]
                pos_ce_sum = np.sum(pos_ce)
                topk_sum = np.sum(mce_tp_srt[:topk])
               
                tp_mu_top_pos[r,c]=topk_sum / pos_ce_sum
                
            if c == 1:
                
                mce_tp_srt = mce_tp_srt[::-1]
                pos_ind = np.where(mce_tp>0)
                
                if len(mce_tp[pos_ind]) > 0:
                    pos_ce = mce_tp[pos_ind]
                    pos_ce_sum = np.sum(pos_ce)
                    topk_sum = np.sum(mce_tp_srt[:topk])
                    
                    tp_mu_top_pos[r,c]=topk_sum / pos_ce_sum
                else:
                    tp_mu_top_pos[r,c]= 0
            
    tp_mu_top_pos = np.mean(tp_mu_top_pos,axis=0)
    topk_meth_pos[m,:]=tp_mu_top_pos
    

pd1 = pd.DataFrame(data=m_eng,columns=['Meth'])
pd2 = pd.DataFrame(topk_meth_pos)
comb = pd.concat([pd1,pd2],axis=1)

f1='...folder/' + ds + \
    'CE_tp_topk_pos_' + typ + '.csv'

comb.to_csv(f1,index=False)
    
   


