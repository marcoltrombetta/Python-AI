#https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge
#https://stackoverflow.com/questions/11361985/output-data-from-all-columns-in-a-dataframe-in-pandas
#http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm
#https://realpython.com/pandas-groupby/

import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df =  pd.read_csv(r'/home/marco/Downloads/covid19/COVID19_line_list_data.csv')
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
'age',
'death',
'country'
]

df = pd.DataFrame(df, columns = custom_paramnames)
#df = df.where((df['death']=='1')  & (df['age']>65))
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df2=df.groupby(['country'])['death'].count()

objects = df2

y_pos = np.array(df2)
performance = df.groupby(['country','death'])['country'].count()
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Death')
plt.title('Programming language usage')

plt.show()
