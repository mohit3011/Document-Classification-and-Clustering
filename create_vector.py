import sys, os
import numpy as np
import copy
import math
from numpy import *
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

Del=0.0000000001

def Transform(data, T=0):
	if(T==0):
		return data  #Identity
	if(T==1):
		return np.log(data+Del)  #Log
	if(T==2):
		return math.sqrt(data)  #Square-Root

def create_word_list():
	file_bbc_terms = open("bbc/bbc.terms", "r")

	words = []
	for row in file_bbc_terms:
		row = row.split("\n")
		words.append(row[0])	# List of words

	return words

def perform_pca(new_train_data, no_samples):

	pca = PCA(n_components = no_samples, svd_solver='full')

	final_train_data = pca.fit_transform(new_train_data)	# final_train_data is the new dataset with PCA applied

	component_array = pca.components_	# Component array 2225 X 9635 , new axes are the linear combination of old axes

	return final_train_data, component_array 


def gmdc(final_train_data, classes):

	GMM = GaussianMixture(n_components = no_samples, covariance_type='diag')
	GMM.fit(final_train_data, classes)

	


def create_classes():

	count_row = 0
	classes = []	# List having the classes for each of the document instance

	file_bbc_classes = open("bbc/bbc.classes", "r")

	for row in file_bbc_classes:
		if count_row > 3:
			row = row.split()
			classes.append(int(row[1]))

		count_row += 1

	return classes

def create_weight_matrix(no_data, no_samples, train_data):

	prob_matrix = np.zeros((no_samples, no_data))

	weight_matrix = np.zeros((no_data))

	for j in range(no_data):
		prob_matrix[:,j] = np.copy(train_data[:,j]/np.sum(train_data[:,j]))

	for j in range(no_data):

		tmp = np.copy(prob_matrix[:,j]*np.log(prob_matrix[:,j]+Del))
		weight_matrix[j] = 1+np.sum(tmp)/np.log(no_samples)

	return weight_matrix

def create_normalized_data(no_data, no_samples, train_data):

	new_train_data = np.zeros((no_samples, no_data))	# Final data matrix

	weight_matrix = create_weight_matrix(no_data, no_samples, train_data)

	for i in range(no_samples):
		
		WordSum=0
		for j in range(no_data):

			new_train_data[i,j] = weight_matrix[j]*Transform(train_data[i,j])

			WordSum+=new_train_data[i,j]*new_train_data[i,j]
		
		WordSum=math.sqrt(WordSum)
		
		new_train_data[i,:]=new_train_data[i,:]/WordSum

	return new_train_data


def create_vectors():
	file_bbc_mtx = open("bbc/bbc.mtx", "r")

	count_row = 0

	no_data=9635
	no_samples=2225

	train_data = np.zeros((no_samples, no_data))	# Sparse data matrix

	for row in file_bbc_mtx:
		if count_row > 1:
			row  = row.split()
			#print row
			train_data[int(row[1])-1][int(row[0])-1] = float(row[2])

		count_row += 1

	new_train_data = create_normalized_data(no_data, no_samples, train_data)

	return new_train_data, no_data, no_samples


if __name__ == '__main__':


	new_train_data, no_data, no_samples = create_vectors()

	words = []	# List of words
	words = create_word_list()

	classes = []	# List having the classes for each of the document instance
	classes = create_classes()

	final_train_data, component_array = perform_pca(new_train_data, no_samples)

	#print final_train_data.shape


