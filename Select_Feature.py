import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def get_data(data_dataframe,window,gap,spec,squeeze):
    N=data_dataframe['Date'].size
    feat_dim=0
    for _,v in spec.items():
        feat_dim += 1 if v==None else len(v)
    data=torch.zeros(N,window,feat_dim,dtype=torch.float)
    cursor=0
    for k,v in spec.items():
        if v==None:
            data[:,0,cursor]=torch.from_numpy(data_dataframe[k].values)
            cursor += 1
        else:
            for idx in v:
                data[:,0,cursor]=torch.FloatTensor([p[idx] for p in data_dataframe[k]])
                cursor += 1
    for w in range(1,window):
        data[w:,w,:]=data[:N-w,0,:]
    if squeeze:
        data=data.view(N,window*feat_dim)
    return data[window-1:-gap],feat_dim

def get_label(data_dataframe,window,gap):
    label=torch.from_numpy(np.array(data_dataframe['Diff'].values>0,dtype=int))
    label=label.type(torch.LongTensor)
    return label[window-1+gap:]


