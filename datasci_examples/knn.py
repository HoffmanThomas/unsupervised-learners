import numpy as np
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
#random seed

random_state = np.random.RandomState(0)
#load the data

data = np.genfromtxt('animalData.csv',delimiter=',',names=True)
#load the data into a numpy array

Xdata=np.array([data['hair'],data['feathers'],data['eggs'],data['milk'],data['airborne'],data['aquatic'],data['predator'],data['toothed'], data['backbone'],data['breathes'],data['venomous'],data['fins'],data['legs'],data['tail'],data['domestic'],data['catsize']]).transpose()
#load yData

yData = np.array(data['type'])
#load the class labels

text_file = open('animalLabels.csv', "r")
#put the class labels into names

names = text_file.read().split('\r')
#create a nearest neighors object using 5 closest neighbors

neighbors = sk.neighbors.NearestNeighbors(n_neighbors=5, algorithm='brute').fit(Xdata)
#find the nearest neighbors and their distances

distances, indices = neighbors.kneighbors(Xdata)
# print out the values
#print indices
#print distances
### uncomment this
#print all the nearest neighbors
for i in range(0,len(indices)):
#get the distances
	dists = distances[i];
##this is the animal we are searching from
	print str(names[i])+'\n'
##this is the animals closest neighbors and their distances
	for j in range(0, len(indices[i])):
		print '\t'+str(j)+' '+str(names[indices[i][j]])+' '+str(dists[j])

names = text_file.read().split('\r')

#create training and test sets, 90% in training 10% in testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xdata, yData, test_size=0.90, random_state=random_state)

#create a nearest neighors classifier using 5 closest neightbors
knnClassifier = KNeighborsClassifier(n_neighbors=50) ### edit this line

#fit the knn model
knnClassifier.fit(X_train, y_train)

#predict the class for the first element of the test set based on neighbors
print(knnClassifier.predict(X_test))

#print accuracy of the knn model on a test set
print(knnClassifier.score(X_test,y_test))


##############
scoutData = np.loadtxt('scoutData.csv',skiprows=1,delimiter=',')
#find the nearest neighbors and their distances to the scoutData

distances, indices = neighbors.kneighbors(scoutData)
#print all the nearest neighbors

for i in range(len(indices[0])):
	print "%2f %s" % (distances[0][i], names[indices[0][i]])
    
