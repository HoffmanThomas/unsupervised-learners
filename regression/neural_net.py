import numpy as np

def logistic(x):
    return 1/(1+np.exp(-x))

print(logistic(-1.3))
