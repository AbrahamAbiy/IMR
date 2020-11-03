import csv as csv
import numpy as np
import pandas as pd

df = pd.read_csv('pneumonia-risk-factors (1).csv', sep=',', encoding='latin-1')

df = df[df.Year != 1991]
df = df[df.Year != 1992]
df = df[df.Year != 1993]
df = df[df.Year != 1994]
df = df[df.Year != 1995]
df = df[df.Year != 1996]
df = df[df.Year != 1997]
df = df[df.Year != 1998]
df = df[df.Year != 1999]
df = df[df.Year != 2000]
df = df[df.Year != 2001]
df = df[df.Year != 2002]
df = df[df.Year != 2003]
df = df[df.Year != 2004]
df = df[df.Year != 2005]
df = df[df.Year != 2006]
df = df[df.Year != 2007]
df = df[df.Year != 2008]
df = df[df.Year != 2009]
df = df[df.Year != 2010]
df = df[df.Year != 2011]
df = df[df.Year != 2012]
df = df[df.Year != 2013]
df = df[df.Year != 2014]
df = df[df.Year != 2015]
df = df[df.Year != 2016]

print(df.tail())

df.to_csv('pn_predictors.csv')
