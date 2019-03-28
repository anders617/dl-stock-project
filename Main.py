import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
from Select_Feature import *

class DNN(nn.Module):
    def __init__(self,feat_dim,hidden_dim,batch_size,num_class,p_drop=0.5):
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
    def __init__(self,feat_dim,hidden_dim,batch_size,num_class,p_drop=0.5):
        super(RNN,self).__init__()
        self.lstm=nn.RNN(feat_dim,hidden_dim,batch_first=True)
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
        hidden_init=torch.zeros(1,self.batch_size,self.hidden_dim)
        return hidden_init

    def forward(self,data_batch):
        _,final_hidden=self.lstm(data_batch,self.hidden)
        return self.fc(final_hidden[0])

class LSTM(nn.Module):
    def __init__(self,feat_dim,hidden_dim,batch_size,num_class,p_drop=0.5):
        super(LSTM,self).__init__()
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
        hidden_init=(torch.zeros(1,self.batch_size,self.hidden_dim),
                     torch.zeros(1,self.batch_size,self.hidden_dim))
        return hidden_init

    def forward(self,data_batch):
        _,(final_hidden,_)=self.lstm(data_batch,self.hidden)
        return self.fc(final_hidden[0])

def predict(data_pred,model,batch_size=None):
    with torch.no_grad():
        model.eval()
        if batch_size==None:
            data_batch=data_pred
            label_pred=torch.argmax(model(data_batch),1)
        else:
            pred_size=len(data_pred)
            label_pred=torch.zeros(pred_size,dtype=torch.long)
            iteration=max(pred_size//batch_size,1)
            acc_array=[0 for i in range(pred_size)]
            for i in range(iteration):
                end=(i+1)*batch_size
                if end>pred_size:
                    end=pred_size
                start=end-batch_size
                data_batch=data_pred[start:end]
                label_pred[start:end]=torch.argmax(model(data_batch),1)
    return label_pred.data.numpy()

def test(data_test,label_test,model,batch_size=None):
    with torch.no_grad():
        label_pred_numpy=predict(data_test,model,batch_size)
    return np.mean(label_pred_numpy==np.array(label_test))

def train(data_train,label_train,data_val,label_val,num_epoch,batch_size,
          learning_rate,model,loss_function,optimizer,random=False,batch_test=False,verbose=True):
    train_size=len(data_train)
    best_acc=0
    for epoch in range(num_epoch):
        iteration_epoch=max(train_size//batch_size,1)
        num_iteration=num_epoch*iteration_epoch
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
                acc=test(data_val,label_val,model,batch_size)
            else:
                acc=test(data_val,label_val,model)
            if acc>best_acc:
                best_acc=acc
                best_model=copy.deepcopy(model)
            if verbose:
                print('Epoch =',epoch+1,', validation accuracy =',acc)
    return best_model

def runProcedure(paras,data_train,label_train,data_test,label_test,random_batch=False,batch_test=True):
    if paras['model_type']=='DNN':
        model=DNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'])
    elif paras['model_type']=='RNN':
        model=RNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'])
    elif paras['model_type']=='LSTM':
        model=LSTM(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'])
    loss_function=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=paras['learning_rate'],eps=1e-08)
    print('Training Phase')
    best_model=train(data_train,label_train,data_train,label_train,paras['num_epoch'],paras['batch_size'],
                     paras['learning_rate'],model,loss_function,optimizer,random_batch,batch_test)
    torch.save(best_model, paras['model_type'])
    print('Testing Phase')
    if not batch_test:
        acc=test(data_test,label_test,best_model)
    else:
        acc=test(data_test,label_test,best_model,paras['batch_size'])
    print('Testing accuracy =',acc)

def main():
    data_dataframe=pd.read_pickle('GSPC_preprocess.pkl')
    spec={'Month':None,'Day':None,'Weekday':None,'VIX':None,
          'Open_n':[-d for d in range(1,40)],
          'High_n':[-d for d in range(1,40)],
          'Low_n':[-d for d in range(1,40)],
          'Close_n':[-d for d in range(1,40)],
          'SMA5_n':[-1],'SMA10_n':[-1],'SMA50_n':[-1],'SMA200_n':[-1],
          'Volume_n':None,'RSI14':None
          }
    paras={'window':10,
           'gap':1,
           'model_type':'LSTM',
           'learning_rate':0.0001,
           'batch_size':10,
           'num_epoch':25,
           'hidden_dim':1024,
           'num_class':2,
           'p_drop':0.5
           }
    for model_type in ['DNN','RNN','LSTM']:
        print('******************** model_type:',model_type,'********************')
        paras['model_type']=model_type
        if paras['model_type']=='DNN':
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,True)
            paras['feat_dim'] *= paras['window']
        else:
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,False)
        label=get_label(data_dataframe,paras['window'],paras['gap'])
        N=label.shape[0]

        data_train=data[:N-400]
        label_train=label[:N-400]
        data_test=data[N-400:]
        label_test=label[N-400:]

        runProcedure(paras,data_train,label_train,data_test,label_test,random_batch=False,batch_test=True)


if __name__=='__main__':
    main()


