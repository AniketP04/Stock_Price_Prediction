#importing lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


#loading data
df=pd.read_csv('/content/RELIANCE.NS.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
del df['Adj Close']

#checking null data
df.isna().sum()

#Scaling data( lstm are very sensitive to the scales)
scaler = StandardScaler()
scaler = scaler.fit(df)
df_s = scaler.transform(df)


#time-steps

def df_X_y(df, window_size=7):
  
  X = []
  y = []
  for i in range(len(df_s)-window_size):
    row = [r for r in df_s[i:i+window_size]]
    X.append(row)
    label = [df_s[i+window_size][3]]
    y.append(label)
  return np.array(X), np.array(y)

X2, y2 = df_X_y(df)
X2.shape, y2.shape

#splitting data into train and test
X2_train, y2_train = X2[:1500], y2[:1500]
X2_val, y2_val = X2[1500:], y2[1500:]

X2_train.shape, y2_train.shape, X2_val.shape, y2_val.shape


#CNN-LSTM model
model = Sequential()


model.add(Conv1D(filters=256, kernel_size= 1, activation='relu', input_shape=(X2_train.shape[1],X2_train.shape[2])))
model.add(Conv1D(filters=128, kernel_size= 1,  activation='relu'))
model.add(MaxPooling1D(pool_size=5, padding='valid'))
model.add(Conv1D(filters=64,  kernel_size= 1, activation='relu'))


model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=75, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))

model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()

model.compile(loss='mean_absolute_error', optimizer='adam')
history=model.fit(X2_train, y2_train, epochs=10,validation_data=(X2_val,y2_val))

yhat=model.predict(X2_val)
yhat

y=yhat.reshape(yhat.shape[0],1)
y.shape


#plt.figure(figsize=(20,8))
#plt.plot(y,color='blue', label='prediction')
#plt.plot(y2_val,color='orange', label='y_test')


test_MAE= mean_squared_error(y2_val, y )


print(f"Test MAE: {test_MAE}")