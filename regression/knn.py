import csv
import numpy as np
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import ensemble

random_state = np.random.RandomState(0) 
data = np.genfromtxt('CRT.csv', delimiter=',', names = True, usecols = (0,1,2,3,4,5,6,7,10)) #this method of import creates a flexible type which does not allow `mean.()` to be called

y_data = np.array([data['StrengthBin']]).transpose()
#y_data = np.array([data['StrengthBin']]).transpose()
print("\nY DATA \n", y_data)

X = np.array([data[x] for x in data.dtype.names])
scaledData = preprocessing.scale(X) #scale the data
mean = scaledData.mean(axis = 0) #mean of scaled data
stdev = scaledData.std(axis = 0) #Std dev of scaled data

data = scaledData

x_data = np.array([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]).transpose()
#x_data = np.array([data['Cement'], data['Slag'], data['FlyAsh'], data['Water'], data['SPlast'], data['CAgg'], data['FAgg'], data['Age']]).transpose()
print("\nX DATA \n", x_data)




# X_train, X_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data, test_size=.1, random_state=5)
# knnClassifier = ensemble.RandomForestClassifier(n_estimators=11,random_state=0).fit(X_train, y_train.ravel())
# #print(knnClassifier.predict(X_test))
# print(knnClassifier.score(X_test,y_test))

max1 = 0
ts = .05
k = 10
opt_k =0
opt_ts = 0
while ts < .90:
	while k < 20:
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


print ('Max Score: ',max1)
print ('Best k for that score: ',opt_k)
print ('Best test size for that score: ',opt_ts)