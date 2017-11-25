import sys, os
import copy
import math
import math
import numpy as np
import pandas as pd
import random as rand
from numpy import *
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import norm
from create_vector import *
from collections import *
from mergingFuntions import *

n_clusters = 5
n_samples = 2225

# lambda_array[i] : ith clusters lambda_i
#


# Likelihood based Merging
def calc_likelihoodChange(cluster1,cluster2):
	pass

# To be updated
def calc_lambda_d(std_cluster_array, n_dim_pca):

	lambda_array = []
	covariance_array = []

	for i in range(n_clusters):
		temp = std_cluster_array[i] * std_cluster_array[i]
		covariance_array.append(temp)

		ans = 1
		for element in temp:
			ans = ans*element
		ans = ans**(float(1)/float(n_dim_pca))
		lambda_array.append(ans)



	return lambda_array, covariance_array



def cal_mean_var(final_train_data, clusters):
    mean_cluster_array = []
    std_cluster_array = []
    prob_cluster = []
    for i in range(n_clusters):
        data_indices = clusters[i]
        data_cluster = final_train_data[data_indices]
        mean_cluster = data_cluster.mean(axis=0)
        std_cluster = data_cluster.std(axis=0)
        mean_cluster_array.append(mean_cluster)
        std_cluster_array.append(std_cluster)
        prob_cluster.append((1.*np.size(clusters[i]))/n_samples)

    return mean_cluster_array,std_cluster_array,np.asarray(prob_cluster)



def create_clusters(kmeans, n_dim_pca):
    clusters = []
    for i in range (n_clusters):
    	clusters.append([])

    for i in range(2225):
    	clusters[kmeans.labels_[i]].append(i)

    mean_vector = np.zeros((n_clusters,n_dim_pca))
    cov_vector = np.zeros((n_clusters,n_dim_pca))
    prob_cluster = np.zeros(n_clusters)
    kmeans_labels = kmeans.labels_
    mean_cluster_array, std_cluster_array, prob_cluster = cal_mean_var(final_train_data,clusters)

    clusters_itr = np.copy(clusters)
    kmeans_labels_itr = np.copy(kmeans_labels)
    prob_cluster_itr = np.copy(prob_cluster)
    mean_cluster_itr = np.copy(mean_cluster_array)
    std_cluster_itr = np.copy(std_cluster_array)

    for i in range(1):
    	mean_cluster_copy = np.copy(mean_cluster_itr)
    	std_cluster_copy = np.copy(std_cluster_itr)
    	updates_labels = np.copy(kmeans_labels_itr)
    	prob_cluster_copy = np.copy(prob_cluster_itr)

    	for k in range(n_samples):
    		find_cluster = updates_labels[k]
    		prob_cluster_max = (-1)*float("inf")
    		feature_k = final_train_data[k]	#Change
    		Prob_k = np.ones(n_clusters)*(-1)*float("inf")

    		for j in range(n_clusters):
    			if np.size(clusters[j])!=1:
    				mu = mean_cluster_copy[j]
    				sig = std_cluster_copy[j]
    				Prob_k[j]=np.log(prob_cluster_copy[j])+np.sum(norm.logpdf(feature_k,mu,sig))
    				if(Prob_k[j] > prob_cluster_max):
    					prob_cluster_max = Prob_k[j]
    					find_cluster = j

    		updates_labels[k] = find_cluster

    	clusters = []

    	for j in range (n_clusters):
    		clusters.append([])

    	for j in range(2225):
    		clusters[updates_labels[j]].append(j)

    	mean_cluster_itr, std_cluster_itr, prob_cluster_itr = cal_mean_var(final_train_data,clusters)
    	kmeans_labels_itr = updates_labels
    print "gmm labels"
    # For functional Merging 
    merge(mean_cluster_itr,std_cluster_itr,prob_cluster_itr,clusters)

    for element in kmeans_labels_itr:
    	print element




if __name__ == '__main__':
	new_train_data, num_dim, num_samples = create_vectors()
	words = []	# List of words
	words = create_word_list()
	y_train = []	# List having the classes for each of the document instance
	y_train = create_classes()
	num_classes = len(np.unique(y_train))
	final_train_data, component_array = perform_pca(new_train_data, num_samples)	#Data after pca
	kmeans = KMeans(n_clusters=n_clusters).fit(final_train_data)	#k-means clustering
	kmeans_labels = kmeans.labels_
	create_clusters(kmeans, num_samples)
	print "k-means labels"
	for element in kmeans_labels:
		print element
