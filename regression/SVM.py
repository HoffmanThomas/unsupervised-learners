import numpy as np
from matplotlib import pylab as pl
from scipy import stats
from sklearn import svm
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import LeaveOneOut

#Load up the labels for the wine ratings  
whitedata = np.genfromtxt('winequality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('/Datasets/winequality-2classlabels-white.csv',delimiter=',',names=True)

#Save the feature names 
names = whitedata.dtype.names

X = np.array([whitedata[names[10]]]).transpose()
Y = np.array(classlabel['poor']).transpose()

X, Y = shuffle(X, Y)

whitedata = np.genfromtxt('winequality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('winequality-2classlabels-white.csv',delimiter=',',names=True)
names = whitedata.dtype.names
random_state = np.random.RandomState(0)

#citric_acid
X = np.array([whitedata[names[0]],whitedata[names[1]],whitedata[names[]]]).transpose()
Y = np.array(classlabel['poor']).transpose()

#90 10
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=random_state)

classifier = svm.SVC( probability=True, random_state=0)
classifier = classifier.fit(X_train,y_train)

predictionSpace = classifier.predict(X_test)
print "Classification score ",classifier.score(X_test, y_test)
probas_ = classifier.predict_proba(X_test)

# get the false positive rate, true positive rate and threshold
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr,tpr) 
print("Area under the ROC curve : %0.4f" % roc_auc)

<<<<<<< HEAD
#machineeeeeeee
=======
#hello world
#goodbye world
#JESSI
>>>>>>> Jessi
