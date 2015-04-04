import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle


dict = pickle.load(open('sample.pickle'));
data_train = dict[0];           
true_label_train = dict[1];     
data_test = dict[2];            
true_label_test = dict[3];      

print("\n------training data-----")
for i in range(data_train.shape[0]):
	print('%s - %s'%(data_train[i],true_label_train[i]))
pass


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

mean_matrix, unq_lbls, global_mean, list_of_ftrs = mean_vector(data_train,true_label_train);

print("\n------mean data-----")
for i in range(mean_matrix.shape[0]):
	print('class %s - %s'%(unq_lbls[i],mean_matrix[i]))
print('global mean vector: %s\n'%(global_mean))

mean_corrected_ftrs_list = []
for i in range(0,len(list_of_ftrs)):
	mean_corrected_ftrs_list.append(list_of_ftrs[i] - global_mean);
pass

print("\n------mean corrected data-----")
for i in range(len(mean_corrected_ftrs_list)):
	print('\nclass %s -\n %s'%(unq_lbls[i],mean_corrected_ftrs_list[i]))


list_classwise_num_of_ftrs = []
for i in range(0,len(mean_corrected_ftrs_list)):
	list_classwise_num_of_ftrs.append(mean_corrected_ftrs_list[i].shape[0])
pass

C_matrix = np.cov(np.transpose(data_train))
print("\n------covar data-----")
print('\n final covariance \n %s'%(C_matrix))
C_inv = np.linalg.inv(C_matrix)
print('\n covariance inverse \n %s'%(C_inv))

List_Prior_Prob = []
for i in range(0,len(list_classwise_num_of_ftrs)):
	List_Prior_Prob.append(1.0*list_classwise_num_of_ftrs[i]/data_train.shape[0]);
pass


print("\n------prior prob data-----")
for i in range(len(List_Prior_Prob)):
	print('\nclass %s -\n %s'%(unq_lbls[i],List_Prior_Prob[i]))
pass


print("\n------testing data-----")
for i in range(data_test.shape[0]):
	print('%s - %s'%(data_test[i],true_label_test[i]))
pass


print("\n ------ test calc -----")
full_list_class_values_test = []
for i in range(0,data_test.shape[0]):
	local_list_class_values =[]
	for j in range(0,len(unq_lbls)):
		q = List_Prior_Prob[j]
		w = 1/(((2*np.pi)**data_train.shape[1])*((np.linalg.det(C_matrix))**0.5))	
		r = reduce(np.dot, [ np.transpose(data_test[i]-global_mean),C_inv,data_test[i]-global_mean])
		x = q*w*np.exp(-0.5*r)
		print ('%f'%x)
		local_list_class_values.append(x)
	pass
	print("--------")
	full_list_class_values_test.append(local_list_class_values)
pass


print("--------")
correct_count = 0;
new_lbl = np.zeros(1)
for i in range(0,len(true_label_test)):
	new_lbl = np.vstack((new_lbl,[np.argmax(full_list_class_values_test[i])+1.0])) 
	if (np.argmax(full_list_class_values_test[i])+1.0 == true_label_test[i]):
		correct_count = correct_count + 1;
	pass
pass
new_lbl = np.delete(new_lbl,(0), axis=0)

print("----final results----")
print ('%d'%correct_count)
print (1.0*correct_count/true_label_test.shape[0])*100


f1 = np.zeros(1)
f2 = np.zeros(1)
f3 = np.zeros(1)
f4 = np.zeros(1)
f5 = np.zeros(1)

for i in range(0,len(full_list_class_values_test)):
	f1 = np.vstack((f1,full_list_class_values_test[i][0]))
pass
f1 = np.delete(f1,(0), axis=0)


for i in range(0,len(full_list_class_values_test)):
	f2 = np.vstack((f2,full_list_class_values_test[i][1]))
pass
f2 = np.delete(f2,(0), axis=0)


for i in range(0,len(full_list_class_values_test)):
	f3 = np.vstack((f3,full_list_class_values_test[i][2]))
pass
f3 = np.delete(f3,(0), axis=0)


for i in range(0,len(full_list_class_values_test)):
	f4 = np.vstack((f4,full_list_class_values_test[i][3]))
pass
f4 = np.delete(f4,(0), axis=0)


for i in range(0,len(full_list_class_values_test)):
	f5 = np.vstack((f5,full_list_class_values_test[i][4]))
pass
f5 = np.delete(f5,(0), axis=0)


# Plot sample data 


colors = iter(cm.rainbow(np.linspace(0,1,5)))
plt.figure(2)
plt.subplot(211)
for i in range(1,6):
    plt.scatter(data_test[(new_lbl==i).reshape((new_lbl==i).size),0],data_test[(new_lbl==i).reshape((new_lbl==i).size),1],color=next(colors))
plt.title('LDA classification plot - 88%')


colors = iter(cm.rainbow(np.linspace(0,1,5)))
plt.subplot(212)
for i in range(1,6):
    plt.scatter(data_test[(true_label_test==i).reshape((true_label_test==i).size),0],data_test[(true_label_test==i).reshape((true_label_test==i).size),1],color=next(colors))
plt.title('Actual classification plot')

plt.show()

xs1 = np.cov(np.transpose(data_train))