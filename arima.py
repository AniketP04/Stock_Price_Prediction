#importing lib
import pandas as pd
import numpy as np
from numpy import diff
import statsmodels.api as sm 
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv(r'/content/RELIANCE.NS.csv',index_col=0)

df.isna().sum()

del df['Volume'],df['Adj Close'],df['Low'],df['High'],df['Open']

#scaling: log-diff
df['close_log'] = np.log(df['Close'])
df['close_log_diff'] = df['close_log']-df['close_log'].shift(1)

#splitting data
train = df[:2000]
val = df[2000:]
train.shape,val.shape


stats.probplot(df['close_log_diff'], dist="norm", plot=plt)

plt.figure(figsize=(16,6))
plt.plot(df['Close'])


#checking stationarity
adfuller(df['close_log_diff'].dropna())

lag_acf = acf(df['close_log_diff'].dropna(), nlags=30)
lag_pacf = pacf(df['close_log_diff'].dropna(), nlags=50, method='ols')

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
pyplot.figure()
pyplot.subplot(211)
plot_acf(df['close_log_diff'].dropna(),ax=pyplot.gca(),lags=40)
plt.title('PACF')
pyplot.subplot(212)
plot_pacf(df['close_log_diff'].dropna(), ax=pyplot.gca(), lags=50)
plt.title('PACF')
pyplot.show()
plt.show()


from pmdarima.arima import auto_arima
smodel = auto_arima(train['Close'], start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=10, max_q=10, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(smodel.summary())

#arima model
from statsmodels.tsa.arima.model import ARIMA


model = ARIMA(train['Close'], order=(2,0,0))  
result = model.fit()
result.summary()


result.plot_diagnostics(figsize=(20, 14))
plt.show()


#predicting test data
fore=result.predict(start = len(train), end = len(df)-1)
fores=pd.Series(fore)

val['d']=fores.values
val['d']

val[['Close','d']].plot(figsize=(12,8))

#predicting for future values
index_future_dates=pd.date_range(start='2023-03-13',end='2023-03-31')
pred=result.predict(start=len(df),end=len(df)+18,).rename('ARIMA Predictions')
pred.index=index_future_dates
print(pred)


pred.plot(figsize=(12,5),legend=True)




#FbProphet model
from prophet import Prophet

df['ds'] = df.index
df['y'] = df['Close']

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=30, freq='D')
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot_components(forecast)


fig = m.plot(forecast, xlabel='Date', ylabel='Price');



