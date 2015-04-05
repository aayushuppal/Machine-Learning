import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
#from mean import function

dict = pickle.load(open('diabetes.pickle'));
data_train = dict[0];           
true_label_train = dict[1];     
data_test = dict[2];            
true_label_test = dict[3];  


plt.scatter(data_train[:, 0], true_label_train[:, 0])
plt.title('Input data')

plt.show()

