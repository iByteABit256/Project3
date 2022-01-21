# C

def reproducibleResults(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def plot_examples(stock_input, stock_decoded):
    n = TIME_STEPS
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, math.ceil(test_samples/TIME_STEPS)))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
         
def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")

TIME_STEPS = 10
window_length = TIME_STEPS
encoding_dim = 3
test_samples = 730

# Commented out IPython magic to ensure Python compatibility.
import math
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from tensorflow import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 113

reproducibleResults(RANDOM_SEED)

df = pd.read_csv("test.csv", sep = "\t", header = None, index_col = 0)

#stock = input('Input stock symbol\n')

input_window = Input(shape=(window_length,1))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x) # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 6 dims
x = Conv1D(16, 2, activation='relu')(x) # 5 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
autoencoder = Model(input_window, decoded)
autoencoder.summary()

#print(X_train[:10][0])
#print(X_test[:10][0])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

global_df = df

for i in range(len(global_df)):

  close = global_df.iloc[i]

  days = pd.date_range(start='1/5/2007', end ='1/1/2017')
  df = pd.DataFrame({'close': close.values}, index = days)

  display(df)

  print("Number of rows and columns:", df.shape)

  split_ind = math.floor(0.80*df.shape[0])
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

  history = autoencoder.fit(
      X_train, X_train,
      epochs=15,
      batch_size=2048,
      shuffle=True,
      validation_data=(X_test, X_test)
  )

  decoded_stocks = encoder.predict(X_test)
  #decoded_stocks = scaler.inverse_transform(decoded_stocks[:,0].reshape(-1,1))

  #plot_results(X_test, decoded_stocks)
  #print(X_test)

  print(decoded_stocks.shape)
  print(test[TIME_STEPS:].close.values.reshape(-1,1).shape)

  plt.figure()
  plt.plot(
    test[TIME_STEPS:].index, 
    scaler.inverse_transform(test[TIME_STEPS:].close.values.reshape(-1,1)), 
    label='close price'
  )

  #print(decoded_stocks[:10,0])
  #print(scaler.inverse_transform(decoded_stocks[:10,0].reshape(-1,1)))

  plt.plot(
    test[TIME_STEPS:].index,
    scaler1.inverse_transform(decoded_stocks[:,0].reshape(-1,1)), 
    label='close price'
  )

  #plot_examples(X_test, decoded_stocks)

  plt.xticks(rotation=25)
  plt.legend()
  plt.savefig("reduce"+str(i)+".png")
