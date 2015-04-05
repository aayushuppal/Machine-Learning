# ordinary linear regression for one attribute

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

inp_data = np.array(([150,6450]
					,[200,7450]
					,[250,8450]
					,[300,9450]
					,[350,11450]
					,[400,15450]
					,[600,18450]
					))

Size_data = inp_data[:,0]*1.0;
Price_data = inp_data[:,1]*1.0;

Y = np.zeros((Price_data.shape[0],1))
Y[:,0] = Price_data[:]

X = np.zeros((Size_data.shape[0],2))
X[:,0] = 1
X[:,1] = Size_data[:]

B = reduce(np.dot, [(np.linalg.inv(reduce(np.dot,[np.transpose(X),X]))),(reduce(np.dot,[np.transpose(X),Y]))])
# B = [b0,b1] here
