import csv
import numpy as np
import sklearn as sk
from sklearn import cross_validation, preprocessing, ensemble

#seed for splitting data into training and testing 
random_state = np.random.RandomState(0) 

#import data using numpy
data = np.genfromtxt('CRT.csv', delimiter=',', names = True, usecols = (0,1,2,3,4,5,6,7,10)) #this method of import creates a flexible type which does not allow `mean.()` to be called

#categories that we fit our data to 
y_data = np.array([data['StrengthBin']]).transpose()

#display the categories which fall into each tuple
print("\nY DATA \n", y_data)

#put data into numpy array for scaling
X = np.array([data[x] for x in data.dtype.names])

#scale the data
scaledData = preprocessing.scale(X) 

#copy scaled data to data
data = scaledData

#interpretation of the data's predictors being used to classify strength 
#x_data = np.array([data['Cement'], data['Slag'], data['FlyAsh'], data['Water'], data['SPlast'], data['CAgg'], data['FAgg'], data['Age']]).transpose()
x_data = np.array([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]).transpose()

#display the data that will predict the strength
print("\nX DATA \n", x_data)

#partition the training and testing data sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=0.90, random_state=random_state)

#n_estimators: Number of trees, generate the random forest
rf = ensemble.RandomForestClassifier(n_estimators=128,random_state=0,bootstrap=True)

#fit the training data to the random forest
rf.fit(X_train,y_train.ravel())

#test our set and observe the accuracy of our classification
print("\nThis model scored: ", rf.score(X_test,y_test))

#store the predictions as an array of 1s and 0s representing high and low
predictions = rf.predict(X_test)

#initialize array for storing text interpretation
readable_pred = []

#convert the 0s and 1s into interpretable data
for item in predictions:
	if item == 1: #if item == 1 it is high strength
		readable_pred.append("H")
	elif item == 0: #if item == 0 it is low strength
		readable_pred.append("L")

#print predictions 
print("\nPredictions:\n", readable_pred)

