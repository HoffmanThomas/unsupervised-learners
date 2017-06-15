import csv
import numpy as np
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import ensemble

#create a random state variable to use with the classifier when getting a random partition point 
random_state = np.random.RandomState(0) 

#pull in the data from CRT.csv and store it in data
data = np.genfromtxt('CRT.csv', delimiter=',', names = True, usecols = (0,1,2,3,4,5,6,7,10)) #this method of import creates a flexible type which does not allow `mean.()` to be called

#set the y_data to be the StrengthBin column 
y_data = np.array([data['StrengthBin']]).transpose()

#print out the y_data we will be trying to match on
print("\nY DATA \n", y_data)

#put the data into a numpy array for scaling 
X = np.array([data[x] for x in data.dtype.names])

scaledData = preprocessing.scale(X) #scale the data
mean = scaledData.mean(axis = 0) #mean of scaled data
stdev = scaledData.std(axis = 0) #Std dev of scaled data

#set the old data to the new scaled data
data = scaledData

#set the x_data to all of the columns besides the y_data
x_data = np.array([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]).transpose()

#display the x_data that we are using 
print("\nX DATA \n", x_data)




# block below loops through all of the training vs. test rations and k values up to max_k to produce the optimal score
max_k = 20
max1 = 0
ts = .05
k = 10
opt_k =0
opt_ts = 0
while ts < .90:
	while k < max_k:
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=ts, random_state=random_state)
		knnClassifier = ensemble.RandomForestClassifier(n_estimators=k,random_state=0).fit(X_train, y_train.ravel())
		#print('\nNum Nearest Neighbors:',k, 'Test Size Split ', ts)
		#print(knnClassifier.predict(X_test))
		score = knnClassifier.score(X_test,y_test)
		if score > max1:
			max1 = score
			opt_k = k
			opt_ts = ts
		k+=1

	k=10	
	ts+=.05

#print out the max score achieved and the parameters used to obtain it
print ('Max Score: ',max1)
print ('Best k for that score: ',opt_k)
print ('Best test size for that score: ',opt_ts)