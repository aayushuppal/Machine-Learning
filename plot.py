import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

data = pickle.load(open("sample.pickle","rb"))
[data0, data1, data2, data3] = data

# Plot sample data 
plt.close("all")

colors = iter(cm.rainbow(np.linspace(0,1,5)))
fig = plt.figure()
for i in range(2,3):
    plt.scatter(data2[(data3==i).reshape((data3==i).size),0],data2[(data3==i).reshape((data3==i).size),1],color=next(colors))
fig.suptitle('Sample data 2')
fig.show()

