from mpl_finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as datetime
import numpy as np
import pandas as pd


def candle_stick(data, dates):
	'''
		args: 
		data (list): list of close prices of sp500 (or S values)
		dates (list): list of dates to go with closing prices (or S values)
	'''

	N = len(data)
	
	fig, ax = plt.subplots()

	quotes = {}
	quotes['open'] = []
	quotes['high'] = []
	quotes['low'] = []
	quotes['close'] = []
	quotes['time'] = []
	for i, value in enumerate(data):
		if i == 0:
			initial_price = value
			continue
		if value > data[i-1]:
			high = value / initial_price
			low = data[i-1] / initial_price
		else:
			low = value / initial_price
			high = data[i-1] / initial_price
		quotes['high'].append(high)
		quotes['low'].append(low)
		quotes['open'].append(data[i-1] / initial_price)
		quotes['close'].append(value / initial_price)
		quotes['time'].append(dates[i])


	candlestick2_ohlc(ax,quotes['open'],quotes['high'],quotes['low'],quotes['close'],width=0.6)

	# This sets how many ticks there are in the plot
	ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
	xdate = [i.astype(str)[:10] for i in quotes['time']]
	#quotes['time'].astype(str)
	#[i.astype(datetime.datetime) for i in quotes['time']]

	def mydate(x,pos):
	    try:
	        return xdate[int(x)]
	    except IndexError:
	        return ''


	ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))

	fig.autofmt_xdate()
	fig.tight_layout()

	plt.show()


def nnet_comparison(data):
	'''
		args: data: list of lists: [[[LSTM], [RNN], [DNN]]]
			where each row is a class with a list of lstm, rnn, and dnn values: [[LSTM], [RNN], [DNN]]
	'''
	fig, axs = plt.subplots()
	
	all_data = []
	i = 1
	xtick_num = []
	xtick_labels = []
	for c, class_data in enumerate(data):
		if c == 0:
			class_num = 2
		elif c == 1:
			class_num = 3
		else:
			class_num = 5
		
		for j, net_data in enumerate(class_data):
			all_data.append(net_data)
			xtick_num.append(i)
			i += 1
			if j == 0:
				xtick_labels.append("LSTM (Class {})".format(class_num))
			elif j == 1:
				xtick_labels.append("RNN (Class {})".format(class_num))
			else:
				xtick_labels.append("DNN (Class {})".format(class_num))

	plt.boxplot(all_data, 0, '')
	plt.xticks(xtick_num,xtick_labels)
	plt.title("Neural Network Performance")
	plt.ylabel("Test Accuracy")
	plt.show()

def main():
	print("Testing plotting")
	
	# data_dataframe=pd.read_pickle('GSPC_preprocess.pkl')
	# N = data_dataframe["Date"].size

	# data = data_dataframe["Close"].values
	# dates = data_dataframe["Date"].values
	# candle_stick(data, dates)

	data = [[[3,2,1], [1,2,5], [5,2,1]], [[10,2,1], [1,5,5], [4,2,1]]]
	nnet_comparison(data)



if __name__=='__main__':
    main()