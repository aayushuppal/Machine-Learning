# ordinary linear regression for multiple attributes datapoints

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

dict = pickle.load(open('diabetes.pickle'));
data_train = dict[0];           
true_label_train = dict[1];     
data_test = dict[2];            
true_label_test = dict[3];

Y = np.zeros((true_label_train.shape[0],1))
Y[:,0] = true_label_train[:,0]

X = np.zeros((data_train.shape[0],data_train.shape[1]+1))
X[:,0] = 1
X[:,1:X.shape[1]] = data_train[:,:]

B = reduce(np.dot, [(np.linalg.inv(reduce(np.dot,[np.transpose(X),X]))),(reduce(np.dot,[np.transpose(X),Y]))])



X2 = np.zeros((data_test.shape[0],data_test.shape[1]+1))
X2[:,0] = 1
X2[:,1:X2.shape[1]] = data_test[:,:]

Y2 = reduce(np.dot,[X2,B])

e = 0;
for i in range(len(Y2)):
	e = e + ((Y2[i]-true_label_test[i])**2)
pass
e = e/len(Y2)
e = np.sqrt(e)