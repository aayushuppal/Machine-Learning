# expectation maximization
import numpy as np
from scipy.stats import norm
import os

os.system('clear')

X = np.array(([2.3,3.2,3.1,1.6,1.9,11.5,10.2,12.3,8.6,10.9]))

pi_1 = 0.5
pi_2 = 0.5
pi = [pi_1,pi_2]

#initialization of (theta)
mu_1 = 0
sig_1 = 1
theta_1 = [mu_1,sig_1]
mu_2 = 0
sig_2 = 1
theta_2 = [mu_2,sig_2]
theta = [theta_1,theta_2]

print "pi_1:",pi_1," pi_2:",pi_2
print "\ninitial (theta)"
print "mu_1:",mu_1," sig_1:",sig_1
print "mu_2:",mu_2," sig_2:",sig_2

# k extends from 0 to 1 > 0 - class1; 1 - class2
# theta extends from 0 to 1 > 0 - class1; 1 - class2

def Rik(i,k):
    a = pi[k]
    
    b = norm.pdf(X[i],theta[k][0],theta[k][1])
    
    c = pi[0]*norm.pdf(X[i],theta[0][0],theta[0][1]) + pi[1]*norm.pdf(X[i],theta[1][0],theta[1][1])
    
    return (a*b)/c




def MuCal():
    xx = 0;
    for i in range(X.shape[0]):
        xx = xx + Rik(i,0)*X[i]
    
    yy = 0
    for i in range(X.shape[0]):
        yy = yy + Rik(i,0)    

    temp_mu1 = xx/yy
    
    
    xx = 0;
    for i in range(X.shape[0]):
        xx = xx + Rik(i,1)*X[i]
    
    yy = 0
    for i in range(X.shape[0]):
        yy = yy + Rik(i,1)    
    
    temp_mu2 = xx/yy
    
    return temp_mu1,temp_mu2


def SigCal():
    xx = 0;
    for i in range(X.shape[0]):
        xx = xx + Rik(i,0)*X[i]*X[i]
    
    yy = 0
    for i in range(X.shape[0]):
        yy = yy + Rik(i,0)    
    
    a,b = MuCal()
    
    zz = a*a

    temp_sig1 = (xx/yy)-zz
    
    
    xx = 0;
    for i in range(X.shape[0]):
        xx = xx + Rik(i,1)*X[i]*X[i]
    
    yy = 0
    for i in range(X.shape[0]):
        yy = yy + Rik(i,1)    
    
    zz = b*b
    
    temp_sig2 = (xx/yy)-zz
    
    return temp_sig1,temp_sig2 
    
def PiCal():
    
    yy = 0
    for i in range(X.shape[0]):
        yy = yy + Rik(i,0) 
    
    temp_pi1 = yy/10
    
    yy = 0
    for i in range(X.shape[0]):
        yy = yy + Rik(i,1) 
    
    temp_pi2 = yy/10
    
    return temp_pi1,temp_pi2
    
def iterationOne():
    temp_mu1,temp_mu2 = MuCal()
    temp_sig1,temp_sig2 = SigCal()
    temp_pi1,temp_pi2 = PiCal()   
    pi[0] = temp_pi1
    pi[1] = temp_pi2
    theta[0][0] = temp_mu1
    theta[0][1] = temp_sig1
    theta[1][0] = temp_mu2
    theta[1][1] = temp_sig2
    
    print theta
    print pi