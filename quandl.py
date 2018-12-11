import pandas as pd
import Quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = Quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']= (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

df= df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))


df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)  #drop incomplete rows after forcast_col shift process

#print(df.head())

X= np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
X_lately = X[-forecast_out:]
X=X[:-forecast_out]

print(len(X_lately))

df.dropna(inplace=True)
y = np.array(df['label'])

y=np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)


#clf = svm.SVR()  #support vector mechine
clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set)



df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
print(df.tail())
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+= one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()