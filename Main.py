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
        self.seq_len = 30
        self.num_layers = 8
        self.lstm=nn.LSTM(feat_dim,hidden_dim,batch_first=True, num_layers=self.num_layers, dropout=p_drop)
        self.fc=nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(p=p_drop),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(p=p_drop),
            nn.Linear(hidden_dim,num_class)
        )
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.hidden=self.init_hidden()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, np.sqrt(2.0 / len(m.weight.size())))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTMCell):
                nn.init.xavier_normal_(m.weight_hh)
                nn.init.xavier_normal_(m.weight_ih)
                nn.init.constant_(m.bias_hh, 0)
                nn.init.constant_(m.bias_ih, 0)

    def init_hidden(self):
        hidden_init=(torch.zeros(self.num_layers,self.batch_size,self.hidden_dim,device=self.device),
                     torch.zeros(self.num_layers,self.batch_size,self.hidden_dim,device=self.device))
        # nn.init.xavier_normal_(hidden_init[0])
        # nn.init.xavier_normal_(hidden_init[1])
        return hidden_init

    def forward(self,data_batch):
        self.hidden = self.init_hidden()
        output,(final_hidden,_)=self.lstm(data_batch,self.hidden)
        # print(output.contiguous()[:, -1:, :].sum(dim=1))
        return self.fc(final_hidden.mean(dim=0))#output[:, -1, :])#output[:, -1:, :].sum(dim=1))#final_hidden[0])

def plot_stats(acc, running_acc, loss, val_acc, val_it, save=False):
    plt.figure(1)
    plt.clf()
    plt.title('Accuracy')
    plt.xlabel('It Number')
    plt.ylabel('Percent')
    plt.plot(acc)
    plt.plot(running_acc)
    plt.plot(val_it, val_acc)

    if save:
        plt.savefig('acc.png')

    plt.pause(0.0001)  # pause a bit so that plots are updated

    plt.figure(2)
    plt.clf()
    plt.title('Loss')
    plt.xlabel('It Number')
    plt.ylabel('Loss')
    plt.plot(loss)

    if save:
        plt.savefig('loss.png')

    plt.pause(0.0001)  # pause a bit so that plots are updated

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

def test(data_test,label_test,model,device,batch_size=None):
    model.eval()
    with torch.no_grad():
        label_pred_numpy=predict(data_test,model,device,batch_size)
    print('Prediction up ratio =',label_pred_numpy.mean())
    print('True up ratio =',label_test.float().mean().item())
    if device.type=='cuda':
        label_test=label_test.cpu()
    return np.mean(label_pred_numpy==np.array(label_test))

def train(data_train,label_train,data_val,label_val,num_epoch,batch_size,
          learning_rate,model,loss_function,optimizer,device,random=False,batch_test=False,verbose=True):
    train_size=len(data_train)
    best_acc=0
    running_mean = 50
    accuracies = []
    running_acc = []
    losses = []
    val_acc = []
    val_it = []
    for epoch in range(num_epoch):
        iteration_epoch=max(train_size//batch_size,1)
        num_iteration=num_epoch*iteration_epoch
        if epoch<5:
            lr_now=0.001
        elif epoch<10:
            lr_now=0.0005
        else:
            lr_now=0.0001
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
            out = model(data_batch)
            preds = torch.argmax(out, dim=1)
            batch_acc = 100 * (preds == label_batch).sum() / float(preds.shape[0])
            running_mean = 0.95 * running_mean + 0.05 * batch_acc.item()
            loss=loss_function(out,label_batch)
            accuracies.append(batch_acc)
            running_acc.append(running_mean)
            losses.append(loss.item())
            loss.backward()
            plot_stats(accuracies, running_acc, losses, val_acc, val_it)
            optimizer.step()
        with torch.no_grad():
            if batch_test:
                acc=test(data_val,label_val,model,device,batch_size)
            else:
                acc=test(data_val,label_val,model,device)
            if acc>best_acc:
                best_acc=acc
                best_model=copy.deepcopy(model)
            if verbose:
                print('Epoch =',epoch+1,', validation accuracy =',acc)
            val_acc.append(acc * 100)
            val_it.append((epoch + 1) * iteration_epoch)
    plot_stats(accuracies, running_acc, losses, val_acc, val_it, save=True)
    return best_model

def runProcedure(paras,data_train,label_train, data_val, label_val, data_test,label_test,device,random_batch=False,batch_test=True):
    if paras['model_type']=='DNN':
        model=DNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'],device).to(device)
    elif paras['model_type']=='RNN':
        model=RNN(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'],device).to(device)
    elif paras['model_type']=='LSTM':
        model=LSTM(paras['feat_dim'],paras['hidden_dim'],paras['batch_size'],paras['num_class'],paras['p_drop'],device).to(device)
    for p in model.parameters():
        # Clip gradients to [-1, 1]
        # By registering a hook, we make sure it is done as the grads are computed instead of at the end
        # This helps with LSTM exploding grad
        p.register_hook(lambda grad: torch.clamp(grad, -paras['grad_clip'], paras['grad_clip']))
    loss_function=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=paras['learning_rate'],eps=1e-08, weight_decay=paras['w_decay'])
    print('Training Phase')
    best_model=train(data_train,label_train,data_val,label_val,paras['num_epoch'],paras['batch_size'],
                     paras['learning_rate'],model,loss_function,optimizer,device,random_batch,batch_test)
    print('Testing Phase')
    if not batch_test:
        acc=test(data_test,label_test,best_model,device)
    else:
        acc=test(data_test,label_test,best_model,device,paras['batch_size'])
    print('Testing accuracy =',acc)

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

def get_label(data_dataframe,window,gap,device):
    label=torch.from_numpy(np.array(data_dataframe['Diff'].values>0,dtype=int))
    label=label.type(torch.LongTensor)
    label=label.to(device)
    return label[window-1+gap:]

def train_validate_test_split(df, df_labels, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(np.arange(0, df.shape[0], 1))
    m = df.shape[0]
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df[perm[:train_end]]
    train_label = df_labels[perm[:train_end]]
    validate = df[perm[train_end:validate_end]]
    val_label = df_labels[perm[train_end:validate_end]]
    test = df[perm[validate_end:]]
    test_label = df_labels[perm[validate_end:]]
    return train, train_label, validate, val_label, test, test_label

def main():
    data_dataframe=pd.read_pickle('GSPC_preprocess.pkl')
    spec={'Open_n':[-d for d in range(1,2)],
          #'High_n':[-d for d in range(1,2)],
          #'Low_n':[-d for d in range(1,2)],
          'Close_n':[-d for d in range(1,2)],
          #'SMA5_n':[-1],'SMA10_n':[-1],'SMA50_n':[-1],'SMA200_n':[-1],
          'Volume_n':None,'RSI14':None
          }
    paras={'window':30,
           'gap':1,
           'model_type':'LSTM',
           'learning_rate': 3e-3,
           'batch_size':250,
           'num_epoch':60,
           'hidden_dim':128,
           'num_class':2,
           'p_drop':0.0,
           'w_decay': 0.0,
           'grad_clip': 10,
           }
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_type in ['LSTM']:
        print('******************** model_type:',model_type,'********************')
        paras['model_type']=model_type
        if paras['model_type']=='DNN':
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,True,device)
            paras['feat_dim'] *= paras['window']
        else:
            data,paras['feat_dim']=get_data(data_dataframe,paras['window'],paras['gap'],spec,False,device)
        label=get_label(data_dataframe,paras['window'],paras['gap'],device)
        N=label.shape[0]

        data_train=data[N-2000:N-400]
        label_train=label[N-2000:N-400]
        data_val = data[N-400:N-200]
        label_val = label[N-400:N-200]
        data_test=data[N-200:]
        label_test=label[N-200:]

        np.random.seed(498)
        data_train, label_train, data_val, label_val, data_test, label_test = train_validate_test_split(data, label, 0.6, 0.2)

        runProcedure(paras,data_train,label_train, data_val, label_val, data_test,label_test,device,random_batch=True,batch_test=True)


if __name__=='__main__':
    main()


