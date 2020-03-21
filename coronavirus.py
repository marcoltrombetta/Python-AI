#https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
#https://stackoverflow.com/questions/11361985/output-data-from-all-columns-in-a-dataframe-in-pandas
#http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm

import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df =  pd.read_csv(r'/data/Python/covid19/COVID19_line_list_data.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

paramnames = [
'id',
'case_in_country',
'reporting date',
'summary',
'location',
'country',
'gender',
'age',
'symptom_onset',
'If_onset_approximated',
'hosp_visit_date',
'exposure_start',
'exposure_end',
'visiting Wuhan',
'from Wuhan',
'death',
'recovered',
'symptom',
'source',
'link'
]

custom_paramnames = [
'id',
'age',
'death'
]

df = pd.DataFrame(df, columns = custom_paramnames)
df = df.where(((df['death']=='0') | (df['death']=='1'))  & (df['age']>65))

forecast_col = 'id'
forecast_out = int(math.ceil(0.01*len(df))) #prediction %

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head(50))
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
