import numpy as np
from scipy import stats
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer


# Data and outputs

#Pull in the data from the csv

datain=np.loadtxt(open("beerdata.csv","rb"), delimiter=",", skiprows=0)

#Set Y to be the first column
y = datain[:,0]-1 # 178x1 vector classifications

#Set X to be the rest of the columns
X = datain[:,1:] # 178x13 matrix of data points

#Use Z-score to normalize the x data
X = stats.zscore(X, axis=0) # normalize the data by feature


#Sets m as the number of rows in X (X.shape[1] would be the number of columns)
num_rows = X.shape[0] # number of data points
num_cols = X.shape[1]

#Pybrain cannot take in a normal array as data to the classifiers so we need to convert our data array to something it can use and then populate it
data = ClassificationDataSet(num_cols)
for i in range(0,num_rows):
    data.addSample(X[i,:],y[i])

#Converts the data to a vector 
data._convertToOneOfMany()

#Split the data into a set proportion between training and testing 
tstdata, trndata = data.splitWithProportion(0.25)

#create our neural net 
net = buildNetwork(data.indim,20,data.outdim, bias=True, outputbias=True, outclass=SigmoidLayer)



#Create the training data
trainer = BackpropTrainer(net, dataset=trndata, learningrate=5, momentum=0.01, weightdecay=0, verbose=True)
# p = net.activateOnDataset( data )
# print (p)

trainer.trainUntilConvergence()