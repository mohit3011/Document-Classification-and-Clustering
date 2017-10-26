import sys, os
import numpy as np
import copy
import math
from numpy import *
from sklearn.decomposition import PCA

Del=0.0000000001

def Transform(data,T=0):
	if(T==0):
		return data  #Identity
	if(T==1):
		return np.log(data+Del)  #Log
	if(T==2):
		return math.sqrt(data)  #Square-Root

file_bbc_mtx = open("bbc/bbc.mtx", "r")

count_row = 0

NoWords=9635
NoData=2225

train_data = np.zeros((NoData, NoWords))	# Sparse data matrix

for row in file_bbc_mtx:
	if count_row > 1:
		row  = row.split()
		#print row
		train_data[int(row[1])-1][int(row[0])-1] = float(row[2])

	count_row += 1

P = np.zeros((NoData, NoWords))

W = np.zeros((NoWords))

new_train_data = np.zeros((NoData, NoWords))	# Final data matrix

for j in range(NoWords):
	P[:,j]=train_data[:,j]/np.sum(train_data[:,j])

for j in range(NoWords):
	tmp=P[:,j]*np.log(P[:,j]+Del)
	W[j]=1+np.sum(tmp)/np.log(NoData)

for i in range(NoData):
	WordSum=0
	for j in range(NoWords):
		new_train_data[i,j]=W[j]*Transform(train_data[i,j])
		WordSum+=new_train_data[i,j]*new_train_data[i,j]
	WordSum=math.sqrt(WordSum)
	new_train_data[i,:]=new_train_data[i,:]/WordSum
		
words = []	# List of words

file_bbc_terms = open("bbc/bbc.terms", "r")

for row in file_bbc_terms:
	row = row.split("\n")
	words.append(row[0])	# List of words

count_row = 0

classes = []	# List having the classes for each of the document instance

file_bbc_classes = open("bbc/bbc.classes", "r")

for row in file_bbc_classes:
	if count_row > 3:
		row = row.split()
		classes.append(int(row[1]))

	count_row += 1

pca = PCA(n_components = NoData, svd_solver='full')

Final_train_data = pca.fit_transform(new_train_data)	# Final_train_data is the new dataset with PCA applied

print train_data[0]
print new_train_data[0]
print Final_train_data[0]
component_array = pca.components_	# Component array 2225 X 9635 , new axes are the linear combination of old axes


'''pca_lambda = pca.explained_variance_

total = 0
for k in range(len(pca_lambda)):
	total += pca_lambda[k]
										
									# Code for finding out the ideal number of components to keep
sum_k = 0
print len(pca_lambda)
for k in range(len(pca_lambda)):
	sum_k += pca_lambda[k]
	if(sum_k/total > 0.95):
		dim = k+1
		break

print dim'''
