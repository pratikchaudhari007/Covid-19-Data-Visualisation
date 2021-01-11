import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


df = pd.read_csv("covid_19_india.csv")

"""# Insights of Data"""

df.head()

try :
    df.drop('Unnamed: 9',axis=1,inplace= True)
    df.shape
except : 
    print('Done')

df.isnull().sum()

df.info()

df.describe()

df.corr()

print(df['Deaths'].unique())

try :
    a=[]
    for i in df['Deaths'].values:
        if '\xa0' in i:
            a.append(int(i.replace("\xa0", '')))
        else:
            a.append(int(i))

    df['Deaths'] = a
    len(a)
except:
    pass

print(df['Deaths'].unique())

df['Deaths'].astype('int64')

df['State/UnionTerritory'].unique()

len(df['State/UnionTerritory'].unique())

def drop_star(df):
    for i in df['State/UnionTerritory'].iteritems():
        if i[1][-3:] == "***":
            df.drop(i[0],inplace=True)
        
drop_star(df)
df['State/UnionTerritory'].unique()

len(df['State/UnionTerritory'].unique())

"""# Data Visualizations"""

df['Cured'].plot(alpha=0.8)
df['Deaths'].plot(alpha=0.3)
df['Confirmed'].plot(alpha=0.5)
plt.show()

df.groupby('State/UnionTerritory')['Confirmed'].plot()
plt.show()
df.groupby('State/UnionTerritory')['Deaths'].plot()
plt.show()

df['Datetime'] = df['Date']+' '+df['Time']

"""* adding both columns for easy time series analysis"""

l = df.groupby('State/UnionTerritory')
current = l.last()

fig ,ax = plt.subplots(figsize= (12,8))
plt.title('Top 10 Contaminated States')
current = current.sort_values("Confirmed",ascending=False)[:10]
p = sns.barplot(ax=ax,x= current.index,y=current['Confirmed'])
p.set_xticklabels(labels = current.index,rotation=90)
p.set_yticklabels(labels=(p.get_yticks()*1).astype(int))
plt.show()

"""* ### Maharashtra being the most contaminated state followed byKarnataka and Andhra Pradesh with approximately equal cases. """

l = df.groupby('State/UnionTerritory')
current = l.last()
current = current.sort_values("Confirmed",ascending=False)

df['Date'].min(), df['Date'].max()

"""# Time Series Analysis For Maharashtra State

"""

Mah = df.loc[df['State/UnionTerritory'] == 'Maharashtra']
Mah.head()

Mah.shape

"""* Checking the data for any null/ missing value"""

Mah.isnull().sum()

Mah.columns



cols=['Sno', 'Time', 'State/UnionTerritory',
       'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured',
       'Deaths']
Mah['Date'] = Mah['Date']+' '+Mah['Time']
Mah.drop(cols, axis=1, inplace=True)
Mah= Mah.sort_values('Date')
Mah.isnull().sum()

Mah.head()

Mah.index

"""### - The initial index is Sr.no so lets change it to ****" Date "****."""

Mah = Mah.groupby('Date')['Confirmed'].sum().reset_index()

Mah = Mah.set_index('Date')
Mah.index = pd.to_datetime(Mah.index)
Mah.index

"""### - Resampling with 'W' means we are taking the weekly data from the whole time period. (Every Sunday)"""

y = Mah['Confirmed'].resample('W').mean()

y.index

y.fillna(method='ffill',inplace=True)
y['2020':]

Mah.plot(figsize=(16, 6))
plt.show()

"""### - The above is initial graph showing the increasing trend and seasonality in the data.

### Now lets plot the Decomposition Plot which shows :
   - orignal data
   - Trend in the data
   - Seasonality 
   - Residual
"""

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, freq = 20, model='additive')
fig = decomposition.plot()
plt.show()

"""### But why do we decompose time series?
###### When we decompose a time series into components, we usually combine the trend and cycle into a single trend-cycle component (sometimes called the trend for simplicity). Often this is done to help improve understanding of the time series, but it can also be used to improve forecast accuracy.

### Types of decomposition :
   - Multiplicative : The components multiply together to make the time series. If you have an increasing trend, the amplitude of seasonal activity increases. Everything becomes more exaggerated.
   - Addative : In an additive time series, the components add together to make the time series.
   
(Here we used Addative)
"""

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

"""(We used SARIMAX)
### -> Seasonal AutoRegressive Integrated Moving Averages
#### One of the methods available in Python to model and predict future points of a time series is known as SARIMAX, which stands for Seasonal AutoRegressive Integrated Moving Averages with eXogenous regressors

### -> What does an Arima model do?
#### Autoregressive Integrated Moving Average Model. An ARIMA model is a class of statistical models for analyzing and forecasting time series data. It explicitly caters to a suite of standard structures in time series data, and as such provides a simple yet powerful method for making skillful time series forecasts.

### -> How to select perfect ARIMA model
#### Rules for identifying ARIMA models. General seasonal models: ARIMA (0,1,1)x(0,1,1) etc. Identifying the order of differencing and the constant: If the series has positive autocorrelations out to a high number of lags (say, 10 or more), then it probably needs a higher order of differencing.
"""

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

"""### -> We choose the one with lowest AIC value from above. In this case we have => ARIMA(0, 1, 1)x(1, 1, 1, 12)7 - AIC:588.9188045652764"""

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

"""### -> Plot on the training data to check how well our model is predicting."""

pred = results.get_prediction(start=pd.to_datetime('2020-08-02'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
plt.legend()
plt.show()

"""### -> Graph showing predicted trends for the next 50 steps."""

pred_uc = results.get_forecast(steps=50)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
plt.legend()
plt.show()

