# -*- coding: utf-8 -*-
from pandas import read_csv
import pandas as pd
import numpy
import matplotlib.pyplot as plt 
dataset = read_csv('chronic_kidney_disease_data.csv', header=None)

def mean(values):
	return sum(values) / float(len(values))
    

#describe dataset
print(dataset.describe())

#show first 10 rows
print(dataset.head(10))

#get rid of row 25
dataset = dataset.drop(dataset.index[25], axis=1)
#replace ? with NaN
dataset[:] = dataset[:].replace('?', numpy.NaN)
#print(dataset.shape)
#get rid of rows missing data
dataset.dropna(inplace=True)
print(dataset.head(10))
print(dataset.describe())
print(dataset.shape)


#print remaining dataset to csv file
#dataset.to_csv('testFile.csv')


#plt.plot()
#plt.show()
