
#Importing pandas library as pd
import pandas as pd

#To import train_test_split 
from sklearn.model_selection import train_test_split

#importing Linear Regression Model
from sklearn.linear_model import LinearRegression

#importing Numpy as np
import numpy as np 

#load the dataset 
score_data = pd.read_csv('/Users/shashankraibole/Downloads/adm_data1.csv')

#splitting the dataset into features(independent) and target(dependent) variables

independent = score_data.iloc[:,0:1]

dependent = score_data.iloc[:,7:8]

#split the dataset into training and testing sets

Xtrain,Xtest,Ytrain,Ytest = train_test_split(independent,dependent,test_size=0.2)

#Declaring lsr as an object of LinearRegression
lsr = LinearRegression()

#Fitting the  with the training datasets
lsr.fit(Xtrain.values,Ytrain.values)

#Predicting the output on the basis of the predict
Ypred =lsr.predict(Xtrain.values)

#Get the GRE score as input from the user
val = input ( 'Enter GRE Score: ')

#Convert the input to an integer
num = int(val)

#Create a numpy array with GRE score
val = np.array ([num])

#Reshape the array to make it compatible for prediction
val = val.reshape (1, -1)

#predicting the output on the basis of numpy array
pred = lsr.predict (val)

#Printing the output
print( ' Chance of admit = ',pred)


