import numpy as np
import pandas as pd
import csv

def preprocessing():
    w_5=5
    w_10=10
    w_50=50
    w_200=200
    w_14=14
    #assets = ['^GSPC.csv', 'FB.csv']
    # 'GOOG','FB','MSFT','TWTR','T','VZ','INTC','AMZN','DIS','HD','XOM','CVX','JNJ','PFE','MRK',
    data_frames = []
    assets = ['^DJI','^IXIC','^RUT','^GSPC']
    for asset in assets:
        data=pd.read_csv('data/' + asset + '.csv')
        #data_vix=pd.read_csv('VIX.csv')
        N=data['Date'].size
        
        # Date
        data['Date']=pd.to_datetime(data['Date'],format="%Y/%m/%d")
        data['Month']=data['Date'].dt.month
        data['Day']=data['Date'].dt.day
        data['Weekday']=data['Date'].dt.weekday

        # VIX
        #data['VIX']=data_vix['Close']/15
        
        # Normalized Volume
        data['Volume_n']=(data['Volume']/data['Volume'].rolling(w_200).mean()-1)*2
        
        # Simple Moving Averages
        data['SMA5']=data['Close'].rolling(w_5).mean()
        data['SMA10']=data['Close'].rolling(w_10).mean()
        data['SMA50']=data['Close'].rolling(w_50).mean()
        data['SMA200']=data['Close'].rolling(w_200).mean()
        
        # Normalized Prices
        Open_n=[[None for j in range(w_200)] for i in range(N)]
        High_n=[[None for j in range(w_200)] for i in range(N)]
        Low_n=[[None for j in range(w_200)] for i in range(N)]
        Close_n=[[None for j in range(w_200)] for i in range(N)]
        for d in range(w_200-1,N):
            Open_n[d]=list((data['Open'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
            High_n[d]=list((data['High'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
            Low_n[d]=list((data['Low'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
            Close_n[d]=list((data['Close'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
        data['Open_n'],data['High_n'],data['Low_n'],data['Close_n']=\
            pd.Series(Open_n),pd.Series(High_n),pd.Series(Low_n),pd.Series(Close_n)
        
        # Normalized Simple Moving Averages
        SMA5_n=[[None for j in range(w_200)] for i in range(N)]
        SMA10_n=[[None for j in range(w_200)] for i in range(N)]
        SMA50_n=[[None for j in range(w_200)] for i in range(N)]
        SMA200_n=[[None for j in range(w_200)] for i in range(N)]
        for d in range(w_200-1,N):
            SMA5_n[d]=list((data['SMA5'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
            SMA10_n[d]=list((data['SMA10'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
            SMA50_n[d]=list((data['SMA50'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
            SMA200_n[d]=list((data['SMA200'][d+1-w_200:d+1]/data['Close'][d]-1)*10)
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
        #print(data2)
        print(asset, data2.shape)
        data_frames.append(data2)


    data2 = pd.concat(data_frames)
    print("End of file: ", data2.shape)
    #Save Data
    data2.to_pickle('all_preprocess.pkl')
    

if __name__=='__main__':
    preprocessing()


