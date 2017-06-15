# Imports
import numpy as np
from scipy import stats
from pybrain.datasets import ClassificationDataSet
# Data and outputs
datain=np.loadtxt(open("beerdata.csv","rb"), delimiter=",", skiprows=0)
y = datain[:,0]-1 # 178x1 vector classifications
X = datain[:,1:] # 178x13 matrix of data points
X = stats.zscore(X, axis=0) # normalize the data by feature
m = X.shape[0] # number of data points
### Add your code here!
  
data = ClassificationDataSet(13)
for i in range(0,X.shape[0]):
    data.addSample(X[i,:],y[i])