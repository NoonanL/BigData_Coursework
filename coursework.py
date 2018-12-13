# -*- coding: utf-8 -*-
from pandas import read_csv
import pandas as pd
import numpy
import seaborn as sns
import matplotlib.pyplot as plt 
dataset = read_csv('chronic_kidney_disease_data.csv', header=0)

# Treats the dataset
# Replaces string values with numeric values for computation
dataset[['htn','dm','cad','pe','ane']] = dataset[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
dataset[['rbc','pc']] = dataset[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
dataset[['pcc','ba']] = dataset[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
dataset[['appet']] = dataset[['appet']].replace(to_replace={'good':1,'poor':0,'no':numpy.nan})
dataset['class'] = dataset['class'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

# Get rid of last column (shouldnt have values here)
dataset.rename({"Unnamed: 25":"a"}, axis="columns", inplace=True)
dataset.drop(["a"], axis=1, inplace=True)

# Replace '?' with NaN
dataset[:] = dataset[:].replace('?', numpy.NaN)
print("Rows missing values: ")
print('rbc     ' + str(dataset['rbc'].isnull().sum()))
print('pc     ' + str(dataset['pc'].isnull().sum()))
print("")

# Make sure the numeric columns are indeed numeric, read_csv seems to init these as objects in some cases
dataset[["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]] = dataset[["age",
"bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]].apply(pd.to_numeric)

# get rid of rows missing data
dataset.dropna(inplace=True)

# The following lines were used to print information about the dataset, uncomment to use:
# print("Rows with ckd " + str((dataset['class'] == 1).sum()))
# print("Rows notckd " + str((dataset['class'] == 0).sum()))
# print("--------------------------------------")
#print remaining dataset to csv file
# dataset.to_csv('testFile.csv')

# for column in dataset:
# 	print("---------------------------------")
# 	print(dataset[column].describe())

# Check for correlations in the dataframe
corr_dataset = dataset.corr()
# Generate a mask for the upper triangle
mask = numpy.zeros_like(corr_dataset, dtype=numpy.bool)
mask[numpy.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(7, 5))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_dataset, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlations between different predictors')
# Uncomment the following line to view the table. Note, this will halt the program at this point.
# plt.show()

# Separate class from the dataset
X = dataset.drop('class',axis=1)
y = dataset['class']

# Scikit function to create a training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Scikit functions to normalise and scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
# apply transformations to the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Scikit Multi-Layer-Perceptron classifier model
from sklearn.neural_network import MLPClassifier
# Define the number of neurons per layer and the number of hidden layers as well as the number of iterations to perform 
mlp = MLPClassifier(hidden_layer_sizes=(25,15,3),max_iter=10000)
# Fit the data to the model
mlp.fit(X_train,y_train)

# Get results using Scikit's predict function
predictions = mlp.predict(X_test)
# Print confusion matrix and classification report to view results
from sklearn.metrics import classification_report,confusion_matrix
print(pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'], margins=True))
print(classification_report(y_test,predictions))
