import csv
import numpy as np
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import ensemble

#set a random state for the classifier to use when it partitions the data into training and testing 
random_state = np.random.RandomState(0) 

#pull in the CSV into the data variable
data = np.genfromtxt('CRT.csv', delimiter=',', names = True, usecols = (0,1,2,3,4,5,6,7,10)) #this method of import creates a flexible type which does not allow `mean.()` to be called

#set the y data to be the StrengthBin 
y_data = np.array([data['StrengthBin']]).transpose()


X = np.array([data[x] for x in data.dtype.names])
scaledData = preprocessing.scale(X) #scale the data
mean = scaledData.mean(axis = 0) #mean of scaled data
stdev = scaledData.std(axis = 0) #Std dev of scaled data

#set our data variable to the scaled data version we just created above
data = scaledData

#set the x data to be the columns other then the y data column 
x_data = np.array([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]).transpose()


#block of code below runs through all test vs. train ratios and k values from 1-max_k to determine the variables to maximize score

max1 = 0
ts = .05
k = 10
opt_k =0
opt_ts = 0
max_k = 20
while ts < .90:
	while k < max_k:
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=ts, random_state=random_state)
		knnClassifier = ensemble.RandomForestClassifier(n_estimators=k,random_state=0).fit(X_train, y_train.ravel())
		score = knnClassifier.score(X_test,y_test)
		if score > max1:
			max1 = score
			opt_k = k
			opt_ts = ts
		k+=1

	k=10	
	ts+=.05

#print out the max score we recieved and the variables used to get it 
print ('Max Score: ',max1)
print ('Best k for that score: ',opt_k)
print ('Best test size for that score: ',opt_ts)