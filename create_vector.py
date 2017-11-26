import sys, os
import numpy as np
import copy
import math
import pdb
from numpy import *
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

Del=0.0000000001

def Transform(data, Type="Log"):
	if(Type=="Identity"):
		return data  #Identity
	if(Type=="Log"):
		return np.log(data+Del)  #Log
	if(Type=="Sqrt"):
		return math.sqrt(data)  #Square-Root

def create_word_list():
	file_bbc_terms = open("bbc/bbc.terms", "r")

	words = []
	for row in file_bbc_terms:
		row = row.split("\n")
		words.append(row[0])	# List of words

	return words

def perform_pca(new_train_data, num_samples):

	pca = PCA(n_components = num_samples, svd_solver='full')

	final_train_data = pca.fit_transform(new_train_data)	# final_train_data is the new dataset with PCA applied

	component_array = pca.components_	# Component array 2225 X 9635 , new axes are the linear combination of old axes

	return final_train_data, component_array


def gmdc_bic_calc(final_train_data):

	count_cluster = 1
	cluster_list = []
	bic_list = []
	bic_list_1 = []
	y_list = []
	for num_clusters in range(1, 10):
		estimator = GaussianMixture(n_components = num_clusters, covariance_type='diag', max_iter=250)
		estimator.fit(final_train_data)
		cluster_list.append(count_cluster)
		count_cluster += 1
		bic_list.append(estimator.bic(final_train_data))
		bic_list_1.append(estimator.bic(final_train_data)+20000000)

	y_list.append(bic_list)	
	y_list.append(bic_list_1)
	draw_graph(cluster_list, y_list, "Number of Clusters", "BIC Values")
	plt.show()
		

def draw_graph(x_val, y_list, x_label, y_label):

	color_list = ["r", "b", "g"]
	count = 0
	#print len(y_list)
	for y_val in y_list:
		#print y_val
		plt.plot(x_val, y_val, color = color_list[count], linewidth=2.0)
		count += 1
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	plt.show()


def create_classes():

	count_row = 0
	classes = []	# List having the classes for each of the document instance

	file_bbc_classes = open("bbc/bbc.classes", "r")

	for row in file_bbc_classes:
		if count_row > 3:
			row = row.split()
			classes.append(int(row[1]))

		count_row += 1

	return np.asarray(classes)

def create_weight_matrix(num_dim, num_samples, train_data, Type="Entropy"):

	weight_matrix = np.zeros((num_dim))

	if Type=="Entropy":
		prob_matrix = np.zeros((num_samples, num_dim))
		for j in range(num_dim):
			prob_matrix[:,j] = np.copy(train_data[:,j]/np.sum(train_data[:,j]))

		for j in range(num_dim):

			tmp = np.copy(prob_matrix[:,j]*np.log(prob_matrix[:,j]+Del))
			weight_matrix[j] = 1+np.sum(tmp)/np.log(num_samples)

	if Type=="Identity":
		for j in range(num_dim):
			weight_matrix[j] = 1

	if Type=="Normal":
		for j in range(num_dim):
			weight_matrix[j] = 1/math.sqrt(np.sum(train_data[:,j]**2))

	if Type=="Gfldf":
		for j in range(num_dim):
			temp = np.copy(train_data[:,j])
			for k in range(num_samples):
				if temp[k]>0:
					temp[k] = 1

			weight_matrix[j] = np.sum(train_data[:,j])/np.sum(temp)


	return weight_matrix

def create_normalized_data(num_dim, num_samples, train_data, Type_weight="Entropy", Type_transform="Log"):

	new_train_data = np.zeros((num_samples, num_dim))	# Final data matrix

	weight_matrix = create_weight_matrix(num_dim, num_samples, train_data, Type_weight)

	for i in range(num_samples):

		WordSum=0
		for j in range(num_dim):

			new_train_data[i,j] = weight_matrix[j]*Transform(train_data[i,j], Type_transform)

			WordSum+=new_train_data[i,j]*new_train_data[i,j]

		WordSum=math.sqrt(WordSum)

		new_train_data[i,:]=new_train_data[i,:]/WordSum

	return new_train_data

def gmdc(final_train_data, y_train, num_classes):

	estimator = GaussianMixture(n_components = num_classes, covariance_type='diag')
	estimator.fit(final_train_data)
	y_train_pred = estimator.predict(final_train_data)

	return y_train_pred

def CalFMWIndex(y_train,y_train_pred,no_classes,no_classes_pred,no_data):
	
	FMWIndexMatrix = np.zeros((no_classes, no_classes_pred))		# FMWIndexMatrix(True classes,Predicted)
	Ni = np.zeros((no_classes, ))		
	Nj = np.zeros((no_classes_pred, ))

	# pdb.set_trace()
	for i in range(no_data):
		FMWIndexMatrix[y_train[i],y_train_pred[i]]+=1

	for i in range(no_classes):
		Ni[i]=np.sum(FMWIndexMatrix[i,:])

	# pdb.set_trace()
	for j in range(no_classes_pred):
		Nj[j]=np.sum(FMWIndexMatrix[:,j])	

	Nic2=0
	Njc2=0
	Nijc2=0

	for i in range(no_classes):
		Nic2+=nc2(Ni[i])
	
	for i in range(no_classes_pred):
		Njc2+=nc2(Nj[i])
	
	for i in range(no_classes):
		for j in range(no_classes_pred):
			Nijc2+=nc2(FMWIndexMatrix[i,j])

	FMWIndex=Nijc2/sqrt(Nic2*Njc2)
	
	return FMWIndex

def nc2(n):
	ans=n*(n-1)/2.0
	return ans

def create_vectors(Type_weight="Entropy", Type_transform="Log"):
	file_bbc_mtx = open("bbc/bbc.mtx", "r")

	count_row = 0

	num_dim=9635
	num_samples=2225

	train_data = np.zeros((num_samples, num_dim))	# Sparse data matrix

	for row in file_bbc_mtx:
		if count_row > 1:
			row  = row.split()
			#print row
			train_data[int(row[1])-1][int(row[0])-1] = float(row[2])

		count_row += 1

	new_train_data = create_normalized_data(num_dim, num_samples, train_data, Type_weight, Type_transform)

	return new_train_data, num_dim, num_samples


if __name__ == '__main__':


	new_train_data, num_dim, num_samples = create_vectors()

	words = []	# List of words
	words = create_word_list()

	y_train = []	# List having the classes for each of the document instance
	y_train = create_classes()
	num_classes = len(np.unique(y_train))

	final_train_data, component_array = perform_pca(new_train_data, num_samples)
	y_train_pred=gmdc(final_train_data,y_train, num_classes)

	FMWIndex=CalFMWIndex(y_train,y_train_pred,num_classes,num_samples)
	# print FMWIndex
	gmdc_bic_calc(final_train_data)
