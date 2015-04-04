import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
#import matplotlib.pyplot as plt
import pickle

dict = pickle.load(open('sample.pickle'));
data_train = dict[0];           # 242X64
true_label_train = dict[1];     # 242X1
data_test = dict[2];            # 200X64
true_label_test = dict[3];      # 200X1


def ldaLearn(X,y):		# X = data_train, y = true_label_train here
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
   
    return means,covmat

def mean_vector(d_tr,l_tr):
	unq_lbls = np.unique(l_tr); # gives all unique labels from training set
	mean_matrix = np.zeros(shape=(unq_lbls.shape[0],d_tr.shape[1]));
	sum_matrix = np.zeros(shape=(d_tr.shape[1]));
	i = 0;
	list_of_ftrs = [];
	for x in unq_lbls:
		tmp_mean = np.zeros(d_tr.shape[1]);
		lcount=0;
		local_ftr = np.zeros(d_tr.shape[1])
		for y in range(0,l_tr.shape[0]):
			if (x == l_tr[y]):
				local_ftr = np.vstack((local_ftr,d_tr[y]))
				lcount = lcount+1;
				for j in range(0,tmp_mean.shape[0]):
					tmp_mean[j] = tmp_mean[j]+d_tr[y][j]
				pass
			pass
		pass
		local_ftr = np.delete(local_ftr,(0), axis=0)
		list_of_ftrs.append(local_ftr);
		sum_matrix = sum_matrix+tmp_mean;
		mean_matrix[i] = tmp_mean/lcount
		i = i+1;	
	pass		
	return mean_matrix,unq_lbls,sum_matrix/data_train.shape[0],list_of_ftrs


mean_matrix,unq_lbls,global_mean,list_of_ftrs = mean_vector(data_train,true_label_train);

for i in range(0,unq_lbls.shape[0]):
	print('mean vector for class %s: %s\n'%(unq_lbls[i],mean_matrix[i]))


print('global mean vector: %s\n'%(global_mean))

mean_corrected_ftrs_list = []
for i in range(0,len(list_of_ftrs)):
	mean_corrected_ftrs_list.append(list_of_ftrs[i] - global_mean);
pass

list_classwise_num_of_ftrs = []
list_classwise_covar = []
for i in range(0,len(mean_corrected_ftrs_list)):
	x = np.dot(np.transpose(mean_corrected_ftrs_list[i]),mean_corrected_ftrs_list[i])/mean_corrected_ftrs_list[i].shape[0]
	list_classwise_num_of_ftrs.append(mean_corrected_ftrs_list[i].shape[0])
	list_classwise_covar.append(x);
pass

C_matrix = np.zeros(shape=(data_train.shape[1],data_train.shape[1]));
for i in range(0,len(list_classwise_covar)):
	C_matrix = C_matrix+list_classwise_covar[i]*list_classwise_num_of_ftrs[i]/data_train.shape[0]
pass


List_Prior_Prob = []
for i in range(0,len(list_classwise_num_of_ftrs)):
	List_Prior_Prob.append(1.0*list_classwise_num_of_ftrs[i]/data_train.shape[0]);
pass

full_list_class_values_test = []
for i in range(0,data_test.shape[0]):
	local_list_class_values =[]
	for j in range(0,len(unq_lbls)):
		x = reduce(np.dot, [mean_matrix[j],np.transpose(C_matrix),data_test[i]]) - reduce(np.dot, [mean_matrix[j],np.transpose(C_matrix),np.transpose(mean_matrix[j])]) + (np.log(List_Prior_Prob[j]))
		#x = reduce(np.dot, [mean_matrix[j],np.transpose(list_classwise_covar[j]),data_test[i]]) - reduce(np.dot, [mean_matrix[j],np.transpose(list_classwise_covar[j]),np.transpose(mean_matrix[j])]) + (np.log(List_Prior_Prob[j]))
		x = np.absolute(x)
		print x

		local_list_class_values.append(x)
	pass
	full_list_class_values_test.append(local_list_class_values)
pass


correct_count = 0;
for i in range(0,len(true_label_test)):
	if (np.argmin(full_list_class_values_test[i])+1.0 == true_label_test[i]):
		correct_count = correct_count + 1;
	pass
pass

print correct_count
print (1.0*correct_count/true_label_test.shape[0])*100


