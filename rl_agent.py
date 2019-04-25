import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import math
import random
from Main import *

#test_count=0

def get_change(data_dataframe,window,gap,device):
    change=data_dataframe['Close'].pct_change(periods=gap).values
    return torch.FloatTensor(change[window-1+gap:],device=device)

def test_rl_agent(data_dataframe,start,end,interval,action_pred,gap,delta,strategy):
    change=data_dataframe['Close'].pct_change(periods=gap).values
    change=change[start+gap:end+gap:interval]
    S=100
    last_action=0
    for i in range(len(change)):
        S += S*strategy[action_pred[i]]*change[i]
        if i>0 and not last_action==strategy[action_pred[i]]:
            S *= delta
        last_action=strategy[action_pred[i]]
    print('   RL agent gains '+str(S-100)+'% during the testing period')
    print('   S&P gains '+str((data_dataframe['Close'][end+gap]/data_dataframe['Close'][start]-1)*100)+'% during the same period')

def predict(data_pred,model,device,batch_size=None):
    with torch.no_grad():
        model.eval()
        if batch_size==None:
            data_batch=data_pred
            action_pred=torch.argmax(model(data_batch),1)
        else:
            pred_size=len(data_pred)
            action_pred=torch.zeros(pred_size,dtype=torch.long,device=device)
            iteration=max(pred_size//batch_size,1)
            for i in range(iteration):
                end=(i+1)*batch_size
                if end>pred_size:
                    end=pred_size
                start=end-batch_size
                data_batch=data_pred[start:end]
                action_pred[start:end]=torch.argmax(model(data_batch),1)
    if device.type=='cpu':
        return action_pred.data.numpy()
    else:
        return action_pred.cpu().data.numpy()

def test(data_test,change_test,model,device,num_action,strategy,batch_size=None):
    #global test_count
    #test_count += 1
    #print(test_count)
    with torch.no_grad():
        action_pred=predict(data_test,model,device,batch_size)
    print('Long ratio:',action_pred.mean())
    S=100
    S_o=100
    for i in range(len(change_test)):
        S += S*strategy[action_pred[i]]*change_test[i]
        S_o += S_o*(strategy[-1]*(change_test[i]>0).float()+strategy[0]*(change_test[i]<0).float())*change_test[i]
        '''
        if test_count==1:
            print('i=',i)
            print(change_test[i],action_pred[i],strategy[action_pred[i]],strategy[action_pred[i]]*change_test[i])
            print(strategy[-1],strategy[0])
            print((change_test[i]>0).float(),(change_test[i]<0).float())
            print((change_test[i]>0).float()*strategy[-1]+(change_test[i]<0).float()*strategy[0])
            print((strategy[-1]*(change_test[i]>0).float()+strategy[0]*(change_test[i]<0).float())*change_test[i])
            print('S=',S,'S_o',S_o)
        '''
    print('   RL agent gains '+str((S-100).item())+'% during the testing period')
    print('   Omniscient agent gains '+str((S_o-100).item())+'% during the testing period')
    print('   S&P gains '+str(((torch.prod(change_test+1)-1)*100).item())+'% during the same period')
    return (S/100).item()

def select_action(data_batch,batch_size,model,eps_start,eps_end,eps_decay,steps_done,device):
    eps_threshold=eps_end+(eps_start-eps_end)*math.exp(-1.*steps_done/eps_decay)
    sample=np.random.choice(2,batch_size,p=[eps_threshold,1-eps_threshold])
    sample=torch.from_numpy(sample)
    sample=sample.to(device)
    action=torch.tensor([random.randint(0,1)],device=device,dtype=torch.uint8)*(sample==0)
    with torch.no_grad():
        action += model(data_batch).argmax(dim=1).type(torch.uint8)*(sample==1)
    return action

def train(data_train,change_train,data_val,change_val,num_epoch,batch_size,num_action,
          learning_sche,eps_sche,model,optimizer,strategy,device,random=False,batch_test=False,verbose=True):
    train_size=len(data_train)
    best_gain=0
    steps_done=0
    eps_start,eps_end,eps_decay=eps_sche
    strategy_torch=torch.FloatTensor([strategy],device=device).repeat(batch_size,1)
    for epoch in range(num_epoch):
        eps_threshold=eps_end+(eps_start-eps_end)*math.exp(-1.*steps_done/eps_decay)
        print('eps',eps_threshold)
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
            change_batch=change_train[batch_mask]
            action=select_action(data_batch,batch_size,model,eps_start,eps_end,eps_decay,steps_done,device).long().view(-1,1)
            loss=F.smooth_l1_loss(model(data_batch).gather(1,action).squeeze(),strategy_torch.gather(1,action).squeeze()*change_batch)
            loss.backward()
            optimizer.step()
            steps_done += 1
        with torch.no_grad():
            if batch_test:
                gain_train=test(data_train,change_train,model,device,num_action,strategy,batch_size)
                gain_val=test(data_val,change_val,model,device,num_action,strategy,batch_size)
            else:
                gain_train=test(data_train,change_train,model,device,num_action)
                gain_val=test(data_val,change_val,model,device,num_action)
            if gain_val>best_gain:
                best_gain=gain_val
                best_model=copy.deepcopy(model)
            if verbose:
                print('Epoch =',epoch+1,', train gain =',gain_train,', val gain =',gain_val)
    return best_model

def runProcedure(paras,data_train,change_train,data_val,change_val,data_test,change_test,strategy,device,random_batch=False,batch_test=True):
    if paras['model_type']=='DNN':
        model=DNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_action'],paras['p_drop'],device).to(device)
    elif paras['model_type']=='RNN':
        model=RNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_action'],paras['p_drop'],device).to(device)
    elif paras['model_type']=='LSTM':
        model=LSTM(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_action'],paras['p_drop'],device).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=paras['learning_sche'][0][1],eps=1e-08)
    print('Training Phase')
    best_model=train(data_train,change_train,data_val,change_val,paras['num_epoch'],paras['batch_size'],paras['num_action'],
                     paras['learning_sche'],paras['eps_sche'],model,optimizer,strategy,device,random_batch,batch_test)
    print('Testing Phase')
    if not batch_test:
        gain=test(data_test,change_test,best_model,device,paras['num_action'],strategy)
    else:
        gain=test(data_test,change_test,best_model,device,paras['num_action'],strategy,paras['batch_size'])
    print('Testing gain =',gain)
    return best_model

def main():
    data_dataframe=pd.read_pickle('GSPC_preprocess.pkl')
    spec={'Open_n':'List','High_n':'List','Low_n':'List','Close_n':'List',
          'SMA5_n':'List','SMA10_n':'List','SMA50_n':'List','SMA200_n':'List',
          'Volume_n':None,'RSI14':None}
    strategy=[-2,-1,0,1,2]
    paras={'window':10,
           'gap':1,
           'model_type':'LSTM',
           'learning_sche':[(0,0.001),(10,0.0005),(20,0.0001)],
           'eps_sche':(0.5,0.02,1200), #eps_start,eps_end,eps_decay
           'batch_size':20,
           'num_epoch':30,
           'hidden_dim':16,
           'num_action':len(strategy),
           'p_drop':0,
           'delta':1
           }
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_type in ['DNN']:
        print('******************** model_type:',model_type,'********************')
        paras['model_type']=model_type
        if paras['model_type']=='DNN':
            data,paras['feat_dim']=get_data_new(data_dataframe,paras['window'],paras['gap'],spec,True,device)
            paras['feat_dim'] *= paras['window']
        else:
            data,paras['feat_dim']=get_data_new(data_dataframe,paras['window'],paras['gap'],spec,False,device)
        change=get_change(data_dataframe,paras['window'],paras['gap'],device)
        N=data.shape[0]

        data_train,change_train,data_val,change_val,data_test,change_test=train_validate_test_split(data,change,0.8,0.1)

        model=runProcedure(paras,data_train,change_train,data_val,change_val,data_test,change_test,strategy,device,random_batch=False,batch_test=True)


if __name__=='__main__':
    main()
