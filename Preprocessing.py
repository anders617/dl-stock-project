import numpy as np
import pandas as pd
import csv

def preprocessing():
    w_5=5
    w_10=10
    w_50=50
    w_200=200
    w_14=14
    data=pd.read_csv('GSPC.csv')
    data_vix=pd.read_csv('VIX.csv')
    N=data['Date'].size
    
    # Date
    data['Date']=pd.to_datetime(data['Date'],format="%Y/%m/%d")
    data['Month']=data['Date'].dt.month
    data['Day']=data['Date'].dt.day
    data['Weekday']=data['Date'].dt.weekday

    # VIX
    data['VIX']=data_vix['Close']/15
    
    # Normalized Volume
    data['Volume_n']=data['Volume']/data['Volume'].rolling(w_200).mean()
    
    # Simple Moving Averages
    data['SMA5']=data['Close'].rolling(w_5).mean()
    data['SMA10']=data['Close'].rolling(w_10).mean()
    data['SMA50']=data['Close'].rolling(w_50).mean()
    data['SMA200']=data['Close'].rolling(w_200).mean()
    
    # Normalized Prices
    Open_n,High_n,Low_n,Close_n=([[None for j in range(w_200)] for i in range(N)],)*4
    for d in range(w_200-1,N):
        Open_n[d]=list(data['Open'][d+1-w_200:d+1]/data['Close'][d])
        High_n[d]=list(data['High'][d+1-w_200:d+1]/data['Close'][d])
        Low_n[d]=list(data['Low'][d+1-w_200:d+1]/data['Close'][d])
        Close_n[d]=list(data['Close'][d+1-w_200:d+1]/data['Close'][d])
    data['Open_n'],data['High_n'],data['Low_n'],data['Close_n']=\
        pd.Series(Open_n),pd.Series(High_n),pd.Series(Low_n),pd.Series(Close_n)
    
    # Normalized Simple Moving Averages
    SMA5_n,SMA10_n,SMA50_n,SMA200_n=\
        ([[None for j in range(w_200)] for i in range(N)],)*4
    for d in range(w_200-1,N):
        SMA5_n[d]=list(data['SMA5'][d+1-w_200:d+1]/data['Close'][d])
        SMA10_n[d]=list(data['SMA10'][d+1-w_200:d+1]/data['Close'][d])
        SMA50_n[d]=list(data['SMA50'][d+1-w_200:d+1]/data['Close'][d])
        SMA200_n[d]=list(data['SMA200'][d+1-w_200:d+1]/data['Close'][d])
    data['SMA5_n'],data['SMA10_n'],data['SMA50_n'],data['SMA200_n']=\
        pd.Series(SMA5_n),pd.Series(SMA10_n),pd.Series(SMA50_n),pd.Series(SMA200_n)
    
    # Relative Strength Index
    data['Diff']=data['Close'].diff()
    data['U']=data['Diff'].clip(0)
    data['D']=(-data['Diff']).clip(0)
    data['RS']=data['U'].rolling(w_14).mean()/data['D'].rolling(w_14).mean()
    data['RSI14']=1-1/(1+data['RS'])

    # Drop first 200
    data2=data.tail(N-w_200).reset_index(drop=True)

    # Save Data
    data2.to_pickle('GSPC_preprocess.pkl')
    

if __name__=='__main__':
    preprocessing()


