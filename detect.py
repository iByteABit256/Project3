import math
import getopt, sys
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from pylab import rcParams
from tensorflow import keras
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 30
THRESHOLD = 0.65

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

dataset = ""
input_n = -1

# Command line arguments

argList = sys.argv[1:]
options = "d:n:m:"

if(len(argList) < 4):
    sys.exit("Input Error\nUsage: -d [dataset] -n [number of lines used] -m [max absolute error]")

try:
    arguments, values = getopt.getopt(argList, options)

    for currArg, currVal in arguments:
        if currArg in ("-d"):
            dataset = currVal
        elif currArg in ("-n"):
            input_n = int(currVal)
        elif currArg in ("-m"):
            THRESHOLD = float(currVal)
        elif currArg in ("-h"):
            print("Usage: -d [dataset] -n [number of lines used] -m [max absolute error]")

except getopt.error as err:
    sys.exit(str(err))

df = pd.read_csv(dataset, sep = "\t", header = None, index_col = 0)

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64, 
    input_shape=( TIME_STEPS , 1)
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=TIME_STEPS))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1)))
model.compile(loss='mae', optimizer='adam')

global_df=df

for i in range(input_n):

	close = global_df.iloc[i]

	days = pd.date_range(start='1/5/2007', end ='1/1/2017')
	df = pd.DataFrame({'close': close.values}, index = days)

	display(df)

	print("Number of rows and columns:", df.shape)

	split_ind = math.floor(0.8*df.shape[0])
	train = df.iloc[:split_ind]
	test = df.iloc[split_ind:]

	print(train.shape)
	print(test.shape)

	scaler = StandardScaler()
	scaler = scaler.fit(train[['close']])
	scaler1 = scaler.fit(test[['close']])

	train['close'] = scaler.transform(train[['close']])
	test['close'] = scaler1.transform(test[['close']])

	# reshape to [samples, time_steps, n_features]

	X_train, y_train = create_dataset(train[['close']], train.close, TIME_STEPS)
	X_test, y_test = create_dataset(test[['close']], test.close, TIME_STEPS)

	print(X_train.shape)

	history = model.fit(
	  X_train, y_train,
	  epochs=10,
	  batch_size=32,
	  validation_split=0.1,
	  shuffle=False
	)

	X_train_pred = model.predict(X_train)

	train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

	X_test_pred = model.predict(X_test)

	test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)


	test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
	test_score_df['loss'] = test_mae_loss
	test_score_df['threshold'] = THRESHOLD
	test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
	test_score_df['close'] = test[TIME_STEPS:].close
	anomalies = test_score_df[test_score_df.anomaly == True]

	anomalies.head()

	plt.figure()

	plt.plot(
	test[TIME_STEPS:].index, 
	scaler.inverse_transform(test[TIME_STEPS:].close.values.reshape(-1,1)), 
	label='close price'
	)

	plt.scatter(
	  anomalies.index,
	  scaler.inverse_transform(anomalies.close.values.reshape(-1,1)),
	  color=sns.color_palette()[3],
	  s=20,
	  label='anomaly'
	)
	plt.xticks(rotation=25)
	plt.legend();
	plt.savefig("detect"+str(i)+".png")
