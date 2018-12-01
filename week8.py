from pandas import read_csv
import pandas
import numpy
dataset = read_csv('chronic_kidney_disease_data.csv', header=None)

# print summary statistics on each attribute 
print(dataset.describe()) 
# print the first 10 rows of data 
#print(dataset.head(10)) 

# count of the number of missing values on each of these columns 
print((dataset[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]] == '?').sum()) 

# mark zero values as missing or NaN 
dataset[[1,2]] = dataset[[1,2]].replace('?', numpy.NaN) 
# count the number of NaN values in each column 
#print(dataset.isnull().sum()) 
# print the first 10 rows of data 
#print(dataset.head(10))

#  remove all rows with missing data 
dataset.dropna(inplace=True) 
# summarize the number of rows and columns in the dataset 
print(dataset.shape) 
