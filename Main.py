import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

class DNN(nn.Module):
    def __init__(self,feat_dim,hidden_dim,batch_size,num_class,p_drop,device):
        super(DNN,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(feat_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim,num_class)
        )

    def forward(self,data_batch):
        return self.fc(data_batch)

class RNN(nn.Module):
    def __init__(self,feat_dim,hidden_dim,batch_size,num_class,p_drop,device):
        super(RNN,self).__init__()
        self.device=device
        self.rnn=nn.RNN(feat_dim,hidden_dim,batch_first=True)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim,num_class)
        )
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.hidden=self.init_hidden()

    def init_hidden(self):
        hidden_init=torch.zeros(1,self.batch_size,self.hidden_dim,device=self.device)
        return hidden_init

    def forward(self,data_batch):
        _,final_hidden=self.rnn(data_batch,self.hidden)
        return self.fc(final_hidden[0])

class LSTM(nn.Module):
    def __init__(self,feat_dim,hidden_dim,batch_size,num_class,p_drop,device):
        super(LSTM,self).__init__()
        self.device=device
        self.lstm=nn.LSTM(feat_dim,hidden_dim,batch_first=True)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim,num_class)
        )
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.hidden=self.init_hidden()

    def init_hidden(self):
        hidden_init=(torch.zeros(1,self.batch_size,self.hidden_dim,device=self.device),
                     torch.zeros(1,self.batch_size,self.hidden_dim,device=self.device))
        return hidden_init

    def forward(self,data_batch):
        _,(final_hidden,_)=self.lstm(data_batch,self.hidden)
        return self.fc(final_hidden[0])

def predict(data_pred,model,device,batch_size=None):
    with torch.no_grad():
        model.eval()
        if batch_size==None:
            data_batch=data_pred
            label_pred=torch.argmax(model(data_batch),1)
        else:
            pred_size=len(data_pred)
            label_pred=torch.zeros(pred_size,dtype=torch.long,device=device)
            iteration=max(pred_size//batch_size,1)
            acc_array=[0 for i in range(pred_size)]
            for i in range(iteration):
                end=(i+1)*batch_size
                if end>pred_size:
                    end=pred_size
                start=end-batch_size
                data_batch=data_pred[start:end]
                label_pred[start:end]=torch.argmax(model(data_batch),1)
    if device.type=='cpu':
        return label_pred.data.numpy()
    else:
        return label_pred.cpu().data.numpy()

def test(data_test,label_test,model,device,num_class,batch_size=None):
    with torch.no_grad():
        label_pred_numpy=predict(data_test,model,device,batch_size)
    #print(np.sum(label_pred_numpy==1)/label_pred_numpy.size)
    print('   Prediction label   --- ',end='')
    for c in range(num_class):
        if c<num_class-1:
            print(str(c)+':'+str(np.sum(label_pred_numpy==c)/label_pred_numpy.size)+', ',end='')
        else:
            print(str(c)+':'+str(np.sum(label_pred_numpy==c)/label_pred_numpy.size))
    print('   True testing label --- ',end='')
    for c in range(num_class):
        if c<num_class-1:
            print(str(c)+':'+str(np.sum(np.array(label_test)==c)/np.array(label_test).size)+', ',end='')
        else:
            print(str(c)+':'+str(np.sum(np.array(label_test)==c)/np.array(label_test).size))
    #print('Prediction up ratio =',label_pred_numpy.mean())
    #print('True up ratio =',label_test.float().mean().item())
    if device.type=='cuda':
        label_test=label_test.cpu()
    return np.mean(label_pred_numpy==np.array(label_test))

def train(data_train,label_train,data_val,label_val,num_epoch,batch_size,num_class,
          learning_sche,model,loss_function,optimizer,device,random=False,batch_test=False,verbose=True):
    train_size=len(data_train)
    best_acc=0
    for epoch in range(num_epoch):
        iteration_epoch=max(train_size//batch_size,1)
        num_iteration=num_epoch*iteration_epoch
        for lr_epoch,lr_rate in learning_sche:
            if epoch>=lr_epoch:
                lr_now=lr_rate
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr']=lr_now
        for i in range(iteration_epoch):
            optimizer.zero_grad()
            model.train()
            if random:
                batch_mask=np.random.choice(train_size,batch_size)
            else:
                end=(i+1)*batch_size
                if end>train_size:
                    end=train_size
                start=end-batch_size
                batch_mask=range(start,end)
            data_batch=data_train[batch_mask]
            label_batch=label_train[batch_mask]
            loss=loss_function(model(data_batch),label_batch)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            if batch_test:
                acc_train=test(data_train,label_train,model,device,num_class,batch_size)
                acc_val=test(data_val,label_val,model,device,num_class,batch_size)
            else:
                acc_train=test(data_train,label_train,model,device,num_class)
                acc_val=test(data_val,label_val,model,device,num_class)
            if acc_val>best_acc:
                best_acc=acc_val
                best_model=copy.deepcopy(model)
            if verbose:
                print('Epoch =',epoch+1,', train acc =',acc_train,', val acc =',acc_val)
    return best_model

def runProcedure(paras,data_train,label_train,data_val,label_val,data_test,label_test,device,random_batch=False,batch_test=True):
    if paras['model_type']=='DNN':
        model=DNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'],device).to(device)
    elif paras['model_type']=='RNN':
        model=RNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'],device).to(device)
    elif paras['model_type']=='LSTM':
        model=LSTM(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'],device).to(device)
    loss_function=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=paras['learning_sche'][0][1],eps=1e-08)
    print('Training Phase')
    best_model=train(data_train,label_train,data_val,label_val,paras['num_epoch'],paras['batch_size'],paras['num_class'],
                     paras['learning_sche'],model,loss_function,optimizer,device,random_batch,batch_test)
    print('Testing Phase')
    if not batch_test:
        acc=test(data_test,label_test,best_model,device,paras['num_class'])
    else:
        acc=test(data_test,label_test,best_model,device,paras['num_class'],paras['batch_size'])
    print('Testing accuracy =',acc)
    return best_model

def get_data(data_dataframe,window,gap,spec,squeeze,device):
    N=data_dataframe['Date'].size
    feat_dim=0
    for _,v in spec.items():
        feat_dim += 1 if v==None else len(v)
    data=torch.zeros(N,window,feat_dim,dtype=torch.float,device=device)
    cursor=0
    for k,v in spec.items():
        if v==None:
            data[:,0,cursor]=torch.from_numpy(data_dataframe[k].values).to(device)
            cursor += 1
        else:
            for idx in v:
                data[:,0,cursor]=torch.FloatTensor([p[idx] for p in data_dataframe[k]]).to(device)
                cursor += 1
    for w in range(1,window):
        data[w:,w,:]=data[:N-w,0,:]
    if squeeze:
        data=data.view(N,window*feat_dim)
    return data[window-1:-gap],feat_dim

def get_label(data_dataframe,window,gap,device,class_spec):
    '''
    label=torch.from_numpy(np.array(data_dataframe['Diff'].values>0,dtype=int))
    label=label.type(torch.LongTensor)
    label=label.to(device)
    return label[window-1+gap:]
    '''
    #change=np.array((data_dataframe['Close'][window-1+gap:]/data_dataframe['Close'][window-1:-gap]-1)*100)
    change=data_dataframe['Close'].pct_change(periods=gap).values*100
    change=change[window-1+gap:]
    label=torch.zeros(len(change),dtype=torch.long,device=device)
    #print(label.shape)
    #print(len(class_spec))
    for c in range(len(class_spec)+1):
        if c==0:
            mask=(change<class_spec[c])
        elif c==len(class_spec):
            mask=(change>=class_spec[c-1])
        else:
            mask=(change>=class_spec[c-1])*(change<class_spec[c])
        idx=np.nonzero(mask)
        #print(idx)
        #print(label[idx].shape)
        label[idx]=c
        #mask=torch.ByteTensor(np.uint8(mask))
        #print(label.masked_select(mask).shape)
        #label.masked_select(mask)=c
    #print(label)
    return label

def simplemap_agent(data_dataframe,start,end,interval,label_pred,gap,delta,strategy):
    change=data_dataframe['Close'].pct_change(periods=gap).values
    #print(change[-15:]*100)
    change=change[start+gap:end+gap:interval]
    #print(label_pred)
    #print(change*100)
    S=100
    last_action=0
    for i in range(len(change)):
        #print(strategy[label_pred[i]])
        S += S*strategy[label_pred[i]]*change[i]
        #print('a=',strategy[label_pred[i]],'c=',change[i])
        #print('S='+str(S))
        if i>0 and not last_action==strategy[label_pred[i]]:
            S *= delta
        last_action=strategy[label_pred[i]]
    print('Simple map agent gains '+str(S-100)+'% during the testing period')
    print('S&P gains '+str((data_dataframe['Close'][end+gap]/data_dataframe['Close'][start]-1)*100)+'% during the same period')

def main():
    data_dataframe=pd.read_pickle('GSPC_preprocess.pkl')
    spec={'Open_n':[-d for d in range(1,30)]+[-d for d in range(30,90,5)],
          'High_n':[-d for d in range(1,30)]+[-d for d in range(30,90,5)],
          'Low_n':[-d for d in range(1,30)]+[-d for d in range(30,90,5)],
          'Close_n':[-d for d in range(1,30)]+[-d for d in range(30,90,5)],
          'SMA5_n':[-1],'SMA10_n':[-1],'SMA50_n':[-1],'SMA200_n':[-1],
          'Volume_n':None,'RSI14':None
          }
    # 2 classes setting
    ''
    class_spec=[0]
    strategy=[-1,1]
    ''
    # 3 classes setting
    '''
    class_spec=[-0.35,0.35]
    strategy=[-1,0,1]
    '''
    # 5 classes setting
    '''
    class_spec=[-0.85,-0.35,0.35,0.85]
    strategy=[-2,-1,0,1,2]
    '''
    paras={'window':10,
           'gap':1,
           'model_type':'LSTM',
           'learning_sche':[(0,0.001),(10,0.0005)],
           'batch_size':20,
           'num_epoch':2,
           'hidden_dim':32,
           'num_class':len(class_spec)+1,
           'p_drop':0.05,
           'delta':0.999
           }  
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_type in ['DNN']:
        print('******************** model_type:',model_type,'********************')
        paras['model_type']=model_type
        if paras['model_type']=='DNN':
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,True,device)
            paras['feat_dim'] *= paras['window']
        else:
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,False,device)
        label=get_label(data_dataframe,paras['window'],paras['gap'],device,class_spec)
        N=label.shape[0]

        data_train=data[N-400:]
        label_train=label[N-400:]
        data_val=data[N-400:]
        label_val=label[N-400:]
        data_test=data[N-400:]
        label_test=label[N-400:]

        model=runProcedure(paras,data_train,label_train,data_val,label_val,data_test,label_test,device,random_batch=False,batch_test=True)
        model.eval()
        label_pred=predict(data_test,model,device,paras['batch_size'])
        print('Trained model')
        simplemap_agent(data_dataframe[paras['window']-1:],N-400,N,1,label_pred,paras['gap'],paras['delta'],strategy)
        print('Omniscient agent')
        simplemap_agent(data_dataframe[paras['window']-1:],N-400,N,1,np.array(label_test),paras['gap'],paras['delta'],strategy)
        

if __name__=='__main__':
    main()


