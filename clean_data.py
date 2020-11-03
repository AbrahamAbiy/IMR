import csv as csv
import numpy as np
import pandas as pd

df = pd.read_csv('Child-Mortality-2019.csv', sep=',', encoding='latin-1')
df = df.iloc[0:]
df = df.iloc[0:-23]

df = df.replace('?', np.NaN)
df[df.isnull().sum(axis=1) < 4]
#df = df.columns[df.columns.str.startswith('unnamed : 3')]
#df = df(df.iloc[::1])

#df = df[df.isnull().sum(axis=1) < 5]

df.to_csv('t.csv')
