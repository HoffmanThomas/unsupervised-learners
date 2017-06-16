import numpy as np
from scipy import stats
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.utilities import percentError

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

tstdata_1d, trndata_1d = data.splitWithProportion(0.25)

#Converts the data to a vector 
data._convertToOneOfMany()


#Split the data into a set proportion between training and testing 
tstdata_temp, trndata_temp = data.splitWithProportion(0.25)

tstdata = ClassificationDataSet(13, 3, nb_classes=3)
for n in range(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(13, 3, nb_classes=3)
for n in range(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

#create our neural net 
net = buildNetwork(data.indim,20,data.outdim, bias=True, outputbias=True, outclass=SigmoidLayer)

#Create the training data
trainer = BackpropTrainer(net, dataset=trndata, learningrate=5, momentum=0.01, weightdecay=0, verbose=True)

#trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )

out = net.activateOnDataset(tstdata).argmax(axis = 1)
#output = np.array([net.activate(x) for x, _ in tstdata]).argmax(axis = 1)

print(100 - percentError(out, tstdata_1d))