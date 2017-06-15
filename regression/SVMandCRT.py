#SVM Example in Python
import csv
import numpy as np
from matplotlib import pylab as pl
from scipy import stats
from sklearn import svm
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import LeaveOneOut
from sklearn import preprocessing


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

#Implement the SVM regression method
classifier = svm.SVC(probability = True, random_state=0)
classifier=classifier.fit(X_train, y_train)

#Creates predictions
predictionSpace=classifier.predict(X_test)
print ("Classification score ",classifier.score(X_test, y_test))
probas_ = classifier.predict_proba(X_test)

# get the false positive rate, true positive rate and threshold
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr,tpr) 
print("Area under the ROC curve : %0.4f" % roc_auc)