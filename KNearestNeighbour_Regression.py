
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt


# Importing the dataset
dataset = pd.read_csv('POQTYDataSet.csv',encoding='unicode_escape')
dataset.dropna(subset = ["Invoice Qty"], inplace=True)
dataset.dropna(subset = ["Supplier"], inplace=True)
dataset.dropna(subset = ["Material Name"], inplace=True)
dataset.dropna(subset = ["Unit"], inplace=True)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

print(X)
#Convert the column into categorical columns
states=pd.get_dummies(X['Supplier'],drop_first=True)
# Drop the state coulmn
X=X.drop('Supplier',axis=1)
# concat the dummy variables
X=pd.concat([X,states],axis=1)


states=pd.get_dummies(X['Material Name'],drop_first=True)
# Drop the state coulmn
X=X.drop('Material Name',axis=1)
# concat the dummy variables
X=pd.concat([X,states],axis=1)


states=pd.get_dummies(X['Unit'],drop_first=True)
# Drop the state coulmn
X=X.drop('Unit',axis=1)
# concat the dummy variables
X=pd.concat([X,states],axis=1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    # Accuracy Score on test dataset

#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()    