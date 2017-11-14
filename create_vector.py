import sys, os
import numpy as np
import copy
import math
import pdb
from numpy import *
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

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


def gmdc(final_train_data, y_train, no_classes, test_data, y_test):

	estimator = GaussianMixture(n_components = no_classes, covariance_type='diag')
	# pdb.set_trace()
	estimator.means_init = np.array([final_train_data[y_train == i].mean(axis=0)
                                    for i in range(no_classes)])

	estimator.fit(final_train_data)
	y_train_pred = estimator.predict(final_train_data)

	# map_index = np.array([np.argmax(np.bincount(y_train[y_train_pred == i])) for i in range(no_classes)])
	# print map_index

	y_test_pred = estimator.predict(test_data)
	y_test_pred_2 = y_test_pred
	# for i in range(len(y_test_pred)):
	# 	y_test_pred_2[i] = map_index[y_test_pred[i]]
	# for i in range(20):
	# 	print y_train[i],y_train_pred[i],y_test_pred_2[i]
	train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
	test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
	# test_accuracy_2 = np.mean(y_test_pred_2.ravel() == y_test.ravel()) * 100

	print "Training Accuracy : ", train_accuracy
	print "Test Accuracy : ", test_accuracy
	# print "Test 2 Accuracy : ",test_accuracy_2

	return y_train_pred,y_test_pred

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
	no_classes = len(np.unique(classes))
	print no_classes

	final_train_data, component_array = perform_pca(new_train_data, no_samples)
	kf = KFold(n_splits=4, random_state=None, shuffle=True)
	# kf.get_n_splits(final_train_data)
	for train_index, test_index in kf.split(final_train_data):
		# pdb.set_trace()
		np.random.shuffle(train_index)
		np.random.shuffle(test_index)
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = final_train_data[train_index], final_train_data[test_index]
		y_train, y_test = classes[train_index], classes[test_index]
		# pdb.set_trace()
		y_train_pred, y_test_pred = gmdc(X_train, y_train, no_classes, X_test, y_test)
		pdb.set_trace()

	#print final_train_data.shape
