#https://www.youtube.com/watch?v=JcI5Vnw0b2c&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=2
#http://www.sc.ehu.es/sbweb/fisica/cursoJava/numerico/regresion/regresion.htm
#http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm

import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key='uV-F3u46yzTiRDfeH3nw'

df = quandl.get('WIKI/GOOGL')
df=df[['Close','Open', 'High', 'Low', 'Volume']]

forecast_col = 'Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) #prediction %
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

#features
X =  np.array(df.drop(['label'],1))
#labels
y = np.array(df['label'])
X = preprocessing.scale(X) #normalized
y=np.array(df['label'])

X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.2)  #correct order x_train, x_test,y_train, y_test
clf= LinearRegression()
clf.fit(X_train, y_train) #training the algorithm
accuracy = clf.score(X_test, y_test) #test

print(accuracy)

# Plot outputs
df.plot(x='Close', y='Open', style='o')
plt.title('Close vs Open')
plt.xlabel('Close')
plt.ylabel('Open')
plt.show()
