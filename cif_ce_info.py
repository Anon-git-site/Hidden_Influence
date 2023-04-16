# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)


root = '.../1_ce_all/cif_CE_all_' 

#ds= 'inat_'
#ds= 'plc_'
ds = 'cif_'
descr = 'imb'
base = 'base_' 

meth = ['rem_','dsm_','eos_']

len_meth = len(meth)
runs = 3
n_cls = 10
dims = np.array([64])

b_tp = np.zeros((runs,n_cls))
b_fp = np.zeros((runs,n_cls))

r_tp = np.zeros((runs,n_cls))
r_fp = np.zeros((runs,n_cls))

d_tp = np.zeros((runs,n_cls))
d_fp = np.zeros((runs,n_cls))

e_tp = np.zeros((runs,n_cls))
e_fp = np.zeros((runs,n_cls))
#######################################
r_diff_tp = np.zeros((runs,n_cls))
r_diff_tp_per = np.zeros((runs,n_cls))

d_diff_tp = np.zeros((runs,n_cls))
d_diff_tp_per = np.zeros((runs,n_cls))

e_diff_tp = np.zeros((runs,n_cls))
e_diff_tp_per = np.zeros((runs,n_cls))
######################################
r_diff_fp = np.zeros((runs,n_cls))
r_diff_fp_per = np.zeros((runs,n_cls))

d_diff_fp = np.zeros((runs,n_cls))
d_diff_fp_per = np.zeros((runs,n_cls))

e_diff_fp = np.zeros((runs,n_cls))
e_diff_fp_per = np.zeros((runs,n_cls))

######################################
r_bdif_fp = np.zeros((runs,n_cls))
r_bdif_fp_per = np.zeros((runs,n_cls))

d_bdif_fp = np.zeros((runs,n_cls))
d_bdif_fp_per = np.zeros((runs,n_cls))

e_bdif_fp = np.zeros((runs,n_cls))
e_bdif_fp_per = np.zeros((runs,n_cls))
######################################
r_b_tp = np.zeros((runs,n_cls))
r_b_tp_per = np.zeros((runs,n_cls))

d_b_tp = np.zeros((runs,n_cls))
d_b_tp_per = np.zeros((runs,n_cls))

e_b_tp = np.zeros((runs,n_cls))
e_b_tp_per = np.zeros((runs,n_cls))
#######
r_m_tp = np.zeros((runs,n_cls))
r_m_tp_per = np.zeros((runs,n_cls))

d_m_tp = np.zeros((runs,n_cls))
d_m_tp_per = np.zeros((runs,n_cls))

e_m_tp = np.zeros((runs,n_cls))
e_m_tp_per = np.zeros((runs,n_cls))

######################################
for r in range(runs):
    f = root + base + str(r) + '.csv'
    b = pd.read_csv(f)
    b = b.to_numpy()
    
    btar = b[:,0]
    bprd = b[:,1]
    bidx = b[:,2]
    blen = b[:,3]
    
    for m in range(len_meth):
        f = root + meth[m] + str(r) + '.csv'
        mth = pd.read_csv(f)
        mth = mth.to_numpy()
        
        mtar = mth[:,0]
        mprd = mth[:,1]
        midx = mth[:,2]
        mlen = mth[:,3]
        
        for c in range(n_cls):  
            print(c)
            bctar = btar[btar==c]
            bcprd = bprd[btar==c]
            bcidx = bidx[btar==c]
            bclen = blen[btar==c]
            
            btar_tp = bctar[bcprd==c]
            bprd_tp = bcprd[bcprd==c]
            bidx_tp = bcidx[bcprd==c]
            blen_tp = bclen[bcprd==c]
            
            btar_fp = bctar[bcprd!=c]
            bprd_fp = bcprd[bcprd!=c]
            bidx_fp = bcidx[bcprd!=c]
            blen_fp = bclen[bcprd!=c]
            
            mctar = mtar[mtar==c]
            mcprd = mprd[mtar==c]
            mcidx = midx[mtar==c]
            mclen = mlen[mtar==c]
            
            mtar_tp = mctar[mcprd==c]
            mprd_tp = mcprd[mcprd==c]
            midx_tp = mcidx[mcprd==c]
            mlen_tp = mclen[mcprd==c]
            
            mtar_fp = mctar[mcprd!=c]
            mprd_fp = mcprd[mcprd!=c]
            midx_fp = mcidx[mcprd!=c]
            mlen_fp = mclen[mcprd!=c]
            
            blen_tp_mu = np.mean(blen_tp)
            blen_fp_mu = np.mean(blen_fp)
            
            mlen_tp_mu = np.mean(mlen_tp)
            mlen_fp_mu = np.mean(mlen_fp)
            
            print('b tp',blen_tp_mu)
            print('b fp',blen_fp_mu)
            print('m tp',mlen_tp_mu)
            print('m fp',mlen_fp_mu)
            
            ################################################
            #tp in base but fp in meth bec not in meth tp
            print('tp',bidx_tp.shape, midx_tp.shape)
            print('fp',bidx_fp.shape, midx_fp.shape)
            diff = np.setdiff1d(bidx_tp, midx_tp)
            print('diff',diff.shape, len(diff))
            
            b_idxs = np.zeros(len(diff))
            b_i = np.zeros(len(diff))
            count=0
            for i in range(len(bidx_tp)):
                if bidx_tp[i] in diff:
                    
                    b_idxs[count]=i
                    b_i[count]=bidx_tp[i]
                    count+=1
            
            if len(diff) > 0:
                b_lens = np.zeros(len(diff))
            else:
                b_lens = np.zeros(1)
            
            count=0
            b_idxs = b_idxs.astype(int)
            for i in range(len(b_idxs)):
                b_lens[count]= blen_tp[b_idxs[i]] 
                count+=1
            
            bt_len = btar_tp.shape[0]
            ###################################################
            
            #fp in base become tp in meth
            
            dif_m = np.setdiff1d(midx_tp, bidx_tp)
            print('difM',dif_m.shape, len(dif_m))
            
            b_idxs_m = np.zeros(len(dif_m))
            b_i_m = np.zeros(len(dif_m))
            
            count=0
            for i in range(len(midx_tp)):
                if midx_tp[i] in dif_m:
                    
                    b_idxs_m[count]=i
                    b_i_m[count]=midx_tp[i]
                    count+=1
            
            if len(dif_m) > 0:
                b_lens_m = np.zeros(len(dif_m))
            else:
                b_lens_m = np.zeros(1)
            
            count=0
            b_idxs_m = b_idxs_m.astype(int)
            
            for i in range(len(b_idxs_m)):
                b_lens_m[count]= mlen_tp[b_idxs_m[i]] 
                count+=1
            #print(b_lens_m)
            print('difm mu',np.mean(b_lens_m))
            print('difm per',len(dif_m) / len(mtar_tp))
            print('mtar',mtar_tp.shape[0])
            
            mt_len = mtar_tp.shape[0]
            #########################
            
            b_idxs_bm = np.zeros(len(dif_m))
            b_i_bm = np.zeros(len(dif_m))
            
            count=0
            for i in range(len(bidx_fp)):
                if bidx_fp[i] in dif_m:
                    
                    b_idxs_bm[count]=i
                    b_i_bm[count]=bidx_fp[i]
                    count+=1
            
            if len(dif_m) > 0:
                b_lens_bm = np.zeros(len(dif_m))
            else:
                b_lens_bm = np.zeros(1)
            
            count=0
            b_idxs_bm = b_idxs_bm.astype(int)
            
            for i in range(len(b_idxs_bm)):
                b_lens_bm[count]= blen_fp[b_idxs_bm[i]] 
                count+=1
            #print(b_lens_m)
            print('difbm mu',np.mean(b_lens_bm))
            print('difbm per',len(dif_m) / len(btar_fp))
            print('mbtar',btar_fp.shape[0])
            
            bfp_len = btar_fp.shape[0]
            #################################################
            dif_i = np.intersect1d(bidx_tp, midx_tp)
            print('difi',dif_i.shape, len(dif_i))
            
            b_idxs_i = np.zeros(len(dif_i))
            b_ii = np.zeros(len(dif_i))
            
            count=0
            for i in range(len(bidx_tp)):
                if bidx_tp[i] in dif_i:
                    
                    b_idxs_i[count]=i
                    b_ii[count]=bidx_tp[i]
                    count+=1
            
            if len(dif_i) > 0:
                b_lens_i = np.zeros(len(dif_i))
            else:
                b_lens_i = np.zeros(1)
            
            count=0
            b_idxs_i = b_idxs_i.astype(int)
            
            for i in range(len(b_idxs_i)):
                b_lens_i[count]= blen_tp[b_idxs_i[i]] 
                count+=1
            
            ##########################
            b_idxs_im = np.zeros(len(dif_i))
            b_iim = np.zeros(len(dif_i))
            
            count=0
            for i in range(len(midx_tp)):
                if midx_tp[i] in dif_i:
                    
                    b_idxs_im[count]=i
                    b_iim[count]=midx_tp[i]
                    count+=1
            
            if len(dif_i) > 0:
                b_lens_im = np.zeros(len(dif_i))
            else:
                b_lens_im = np.zeros(1)
            
            count=0
            b_idxs_im = b_idxs_im.astype(int)
            
            for i in range(len(b_idxs_im)):
                b_lens_im[count]= mlen_tp[b_idxs_im[i]] 
                count+=1
            
            ##############
            
            ##############################################
            if m == 0:
                b_tp[r,c] = blen_tp_mu
                b_fp[r,c] = blen_fp_mu
                r_tp[r,c] = mlen_tp_mu
                r_fp[r,c] = mlen_fp_mu
                r_diff_tp[r,c] = np.mean(b_lens)
                r_diff_tp_per[r,c] = len(diff) / bt_len 
                r_diff_fp[r,c] = np.mean(b_lens_m)
                r_diff_fp_per[r,c] = len(dif_m) / mt_len
                r_bdif_fp[r,c] = np.mean(b_lens_bm)
                r_bdif_fp_per[r,c] = len(dif_m) / bfp_len
                
                r_b_tp[r,c] = np.mean(b_lens_i)
                r_b_tp_per[r,c] = len(dif_i) / bt_len
                r_m_tp[r,c] = np.mean(b_lens_im)
                r_m_tp_per[r,c] = len(dif_i) / mt_len
                
            if m == 1:
                d_tp[r,c] = mlen_tp_mu
                d_fp[r,c] = mlen_fp_mu
                d_diff_tp[r,c] = np.mean(b_lens)
                d_diff_tp_per[r,c] = len(diff) / bt_len
                d_diff_fp[r,c] = np.mean(b_lens_m)
                d_diff_fp_per[r,c] = len(dif_m) / mt_len
                d_bdif_fp[r,c] = np.mean(b_lens_bm)
                d_bdif_fp_per[r,c] = len(dif_m) / bfp_len
                
                d_b_tp[r,c] = np.mean(b_lens_i)
                d_b_tp_per[r,c] = len(dif_i) / bt_len
                d_m_tp[r,c] = np.mean(b_lens_im)
                d_m_tp_per[r,c] = len(dif_i) / mt_len
                
            if m == 2:
                e_tp[r,c] = mlen_tp_mu
                e_fp[r,c] = mlen_fp_mu
                e_diff_tp[r,c] = np.mean(b_lens)
                e_diff_tp_per[r,c] = len(diff) / bt_len
                e_diff_fp[r,c] = np.mean(b_lens_m)
                e_diff_fp_per[r,c] = len(dif_m) / mt_len
                e_bdif_fp[r,c] = np.mean(b_lens_bm)
                e_bdif_fp_per[r,c] = len(dif_m) / bfp_len
                
                e_b_tp[r,c] = np.mean(b_lens_i)
                e_b_tp_per[r,c] = len(dif_i) / bt_len
                e_m_tp[r,c] = np.mean(b_lens_im)
                e_m_tp_per[r,c] = len(dif_i) / mt_len
                
            print()


b_tp_mu = np.mean(b_tp,axis=0) / dims
b_fp_mu = np.mean(b_fp,axis=0) / dims

r_tp_mu = np.mean(r_tp,axis=0) / dims
r_fp_mu = np.mean(r_fp,axis=0) / dims

d_tp_mu = np.mean(d_tp,axis=0) / dims
d_fp_mu = np.mean(d_fp,axis=0) / dims

e_tp_mu = np.mean(e_tp,axis=0) / dims
e_fp_mu = np.mean(e_fp,axis=0) / dims

print('btp mu',b_tp_mu)
print('bfp mu',b_fp_mu)
print()
print('rtp mu',r_tp_mu)
print('rfp mu',r_fp_mu)
print()
print('dtp mu',d_tp_mu)
print('dfp mu',d_fp_mu)
print()
print('etp mu',e_tp_mu)
print('efp mu',e_fp_mu)

pd1 = pd.DataFrame(data=b_tp_mu,columns=['Base_TP'])
pd3 = pd.DataFrame(data=r_tp_mu,columns=['REM_TP'])
pd5 = pd.DataFrame(data=d_tp_mu,columns=['DSM_TP'])
pd7 = pd.DataFrame(data=e_tp_mu,columns=['EOS_TP'])

comb=pd.concat([pd1,pd3,pd5,pd7],axis=1)
comb.shape 
f = '...file/' + ds +'CE_tp_fp_comp_' \
    + descr +'.csv'
comb.to_csv(f,index=False)




