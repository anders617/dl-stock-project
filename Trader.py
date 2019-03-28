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

def test_with_100_grand():
    cash = 100000
    data_dataframe=pd.read_pickle('GSPC_preprocess.pkl')

    data_dataframe=pd.read_pickle('GSPC_preprocess.pkl')
    spec={'Month':None,'Day':None,'Weekday':None,
          'VIX':None,
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
    for model_type in ['DNN']:#,'RNN']:#,'LSTM']:
        print('******************** model_type:',model_type,'********************')
        model = torch.load(model_type+".pt")
        model.eval()
        paras['model_type']=model_type
        if paras['model_type']=='DNN':
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,True)
            paras['feat_dim'] *= paras['window']
        else:
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,False)

        label=get_label(data_dataframe,paras['window'],paras['gap'])
        N=label.shape[0]
        print(N)
        print(data.shape)
        data_test=data[N-400:]
        # for i, data_inst in enumerate(data_test):
        #     print("todo")
        predictions = predict(data_test, model)
        
        opening_prices = data_dataframe['Open'][-400:]
        closing_prices = data_dataframe['Close'][-400:]
        for i, day in enumerate(predictions):
            #print(opening_prices.iloc[i])
            number_of_shares = int(cash / opening_prices.iloc[i])
            cash += number_of_shares * (closing_prices.iloc[i] - opening_prices.iloc[i])
    print(cash)
    roi = 100 * (cash / 100000 - 1)
    print(roi)
    return 



def main():
    print("Running Trader.py")
    test_with_100_grand()

if __name__=='__main__':
    main()