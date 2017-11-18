import sys, os
import numpy as np
import copy
import math
from numpy import *
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pandas as pd
import random as rand
from sklearn.cluster import KMeans
from scipy.stats import norm

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

n_data_pca = 2225
pca = PCA(n_components = n_data_pca, svd_solver='full')

Final_train_data = pca.fit_transform(new_train_data)	# Final_train_data is the new dataset with PCA applied

#print train_data[0]
#print new_train_data[0]
#print Final_train_data[0]
component_array = pca.components_	# Component array 2225 X 9635 , new axes are the linear combination of old axes
n_clusters = 5


kmeans = KMeans(n_clusters=n_clusters).fit(Final_train_data)

kmeans_labels = kmeans.labels_

x = np.zeros(n_clusters)
for i in kmeans_labels:
	x[i]=x[i]+1

for i in range(n_clusters):
	print i,x[i]

clusters = []
for i in range (n_clusters):
	clusters.append([])

for i in range(2225):
	clusters[kmeans.labels_[i]].append(i)

mean_vector = np.zeros((n_clusters,n_data_pca))
cov_vector = np.zeros((n_clusters,n_data_pca))
p = np.zeros(n_clusters)
kmeans_labels = kmeans.labels_

for i in range(n_clusters):
	for k in range(n_data_pca):
		vec_attr = []
		for j in range(np.size(clusters[i])):
			vec_attr.append(Final_train_data[clusters[i][j]][k])
		vec_attr = np.array(vec_attr)
		#print np.shape(vec_attr)
		mean_vector[i][k]=vec_attr.mean()
		cov_vector[i][k]=vec_attr.std()
	p[i] = (1.*np.size(clusters[i]))/2225

clusters_itr = np.copy(clusters)
cluster = np.copy(kmeans_labels)
p_x = np.copy(p)
mean_vector_x = np.copy(mean_vector)
cov_vector_x = np.copy(cov_vector)

print "start"

for i in range(10):
	mean_vector_copy = np.copy(mean_vector_x)
	cov_vector_copy = np.copy(cov_vector_x)
	updates = np.copy(cluster)
	for k in range(2225):
		find_cluster = 0
		prob_cluster = float("inf")
		f = Final_train_data[k]
		p_copy = np.copy(p_x)
		for j in range(n_clusters):
			if np.size(clusters[j])!=1:
				mu = mean_vector_copy[j]
				sig = cov_vector_copy[j]
				#if k==1:
				#	print p_copy[j], prob_cluster, j
				p_copy[j]=-1*((np.log(p_copy[j]))+(np.sum(norm.logpdf(f,mu,sig))))
				if(p_copy[j] < prob_cluster):
					prob_cluster = p_copy[j]
					find_cluster = j
		updates[k] = find_cluster
	clusters = []
	mean_vector_x = np.zeros((n_clusters,n_data_pca))
	cov_vector_x = np.zeros((n_clusters,n_data_pca))
	p_x = np.zeros(n_clusters)

	for j in range (n_clusters):
		clusters.append([])

	for j in range(2225):
		clusters[updates[j]].append(j)

	for j in range(n_clusters):
		for k in range(n_data_pca):
			vec_attr = []
			for l in range(np.size(clusters[j])):
				vec_attr.append(Final_train_data[clusters[j][l]][k])
			vec_attr = np.array(vec_attr)
			mean_vector_x[j][k]=vec_attr.mean()
			cov_vector_x[j][k]=vec_attr.std()
		p_x[j] = (1.*np.size(clusters[j]))/2225

	p_copy = p_x
	mean_vector_copy = mean_vector_x
	cov_vector_copy = cov_vector_x
	cluster = updates


a = np.zeros(n_clusters)

for i in range(2225):
	a[cluster[i]]+=1
	print cluster[i]

for i in range(n_clusters):
	print a[i]

#for i in range(100):
#	cluster_points = Final_train_data[]
rand.seed(42)

"""lam = np.random.rand(2225)
lam = lam / (sum(lam))

params = []
for i in range(2225):
	params_i = [];
	mu = (10)*np.random.rand()
	cov = np.random.rand(2225)
	params_i.append(mu)
	params_i.append(cov)
	params_i.append(lam[i])
	params.append(params_i)

shift = 1000
iters = 0


while shift > 0.01:
	iters += 1"""


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
