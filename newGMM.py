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
from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.stats import norm
from create_vector import *
from collections import *
from mergingFuntions import *
from log_likelihood_merge import *
from itertools import permutations

n_clusters = 5			
n_samples = 2225
n_dim_pca = 30
Kmeans_n_clusters=100
n_EM_Update=200
n_mergingIteration=Kmeans_n_clusters-n_clusters

def permute_labels(y_train, y_predict, num_clusters, num_clusters_pred):   #This function computes all the permutations on cluster labels
    label_list = []
    for i in range(num_clusters_pred):
        label_list.append(i)

    max_count = -1
    final_map = ()
    permute_label_list = list(permutations(label_list))
    for permute_label in permute_label_list:
        mapping = ()
        mapping = np.asarray(permute_label)
        New_y_predict = mapping[y_predict]

        count_correct = CalFMWIndex(y_train,New_y_predict,num_clusters,num_clusters_pred,n_samples)
        if count_correct > max_count:
            max_count = count_correct   #max count has the max_accuracy
            final_map = mapping         #final_map has the ideal mapping
    return final_map, max_count

def No_permute_labels(y_train, y_predict, num_clusters, num_clusters_pred):   #This function computes all the permutations on cluster labels
    count_correct = CalFMWIndex(y_train,y_predict,num_clusters,num_clusters_pred,n_samples)
    return count_correct

def calc_likelihoodChange(cluster1,cluster2):
	pass

# To be updated
def calc_lambda_d(std_cluster_array, n_dim_pca,n_itr_clusters):

    lambda_array = []
    covariance_array = []
    for i in range(n_itr_clusters):
        temp = std_cluster_array[i] * std_cluster_array[i]
        ans = 0
        for element in temp:
            ans += math.log(element)

        temp_lambda = (float(ans)/float(n_dim_pca))
        covariance_array.append(temp/math.exp(temp_lambda))
        lambda_array.append(math.exp(temp_lambda))
    # pdb.set_trace()
    return lambda_array, covariance_array

def cal_mean_var(final_train_data, clusters,n_itr_clusters):
    mean_cluster_array = []
    std_cluster_array = []
    prob_cluster = []
    for i in range(n_itr_clusters):
        data_indices = clusters[i]
        data_cluster = final_train_data[data_indices]
        try :
            mean_cluster = data_cluster.mean(axis=0)
        except :
            mean_cluster = 0.0
            print "Error Occured"
        try :
            std_cluster = data_cluster.std(axis=0)
        except :
            std_cluster = 0.0
            print "Error Occured"
        mean_cluster_array.append(mean_cluster)
        std_cluster_array.append(std_cluster)
        prob_cluster.append((1.*np.size(clusters[i]))/n_samples)

    return mean_cluster_array,std_cluster_array,np.asarray(prob_cluster)

def create_clusters(kmeans, n_samples, n_dim_pca, n_itr_clusters, y_train):
    clusters = []
    Accuracy = []
    Map = []
    for i in range(n_itr_clusters):
        clusters.append([])

    for i in range(n_samples):
        clusters[kmeans.labels_[i]].append(i)

    mean_vector = np.zeros((n_itr_clusters,n_dim_pca))
    cov_vector = np.zeros((n_itr_clusters,n_dim_pca))
    prob_cluster = np.zeros(n_itr_clusters)
    kmeans_labels = kmeans.labels_
    mean_cluster_array, std_cluster_array, prob_cluster = cal_mean_var(final_train_data,clusters,n_itr_clusters)

    print "k-means done"

    for M in range(n_mergingIteration):

        print M
        clusters_itr = np.copy(clusters)
        kmeans_labels_itr = np.copy(kmeans_labels)
        prob_cluster_itr = np.copy(prob_cluster)
        mean_cluster_itr = np.copy(mean_cluster_array)
        std_cluster_itr = np.copy(std_cluster_array)

        for i in range(n_EM_Update):
            mean_cluster_copy = np.copy(mean_cluster_itr)
            std_cluster_copy = np.copy(std_cluster_itr)
            updates_labels = np.copy(kmeans_labels_itr)
            prob_cluster_copy = np.copy(prob_cluster_itr)

        for k in range(n_samples):
            find_cluster = updates_labels[k]
            prob_cluster_max = (-1)*float("inf")
            feature_k = final_train_data[k]	#Change
            Prob_k = np.ones(n_itr_clusters)*(-1)*float("inf")

            for j in range(n_itr_clusters):
                if np.size(clusters[j])!=1:
                    mu = mean_cluster_copy[j]
                    sig = std_cluster_copy[j]
                    Prob_k[j]=np.log(prob_cluster_copy[j])+np.sum(norm.logpdf(feature_k,mu,sig))
                    if(Prob_k[j] > prob_cluster_max):
                        prob_cluster_max = Prob_k[j]
                        find_cluster = j

                updates_labels[k] = find_cluster

        clusters = []

        for j in range (n_itr_clusters):
            clusters.append([])
        for j in range(n_samples):
            clusters[updates_labels[j]].append(j)

        mean_cluster_itr, std_cluster_itr, prob_cluster_itr = cal_mean_var(final_train_data,clusters,n_itr_clusters)
        kmeans_labels_itr = updates_labels
        print "EM Update done"
        kmeans_labels = kmeans_labels_itr
        # print mean_cluster_itr, std_cluster_itr, prob_cluster_itr
        lambda_array, covar_D = calc_lambda_d(std_cluster_itr,n_dim_pca,n_itr_clusters)
        print "D & lambda Done"
        clusters,kmeans_labels,n_itr_clusters = merge_log(mean_cluster_itr,std_cluster_itr,prob_cluster_itr,lambda_array,covar_D,clusters,kmeans_labels,n_itr_clusters,n_samples,n_dim_pca)
        # clusters,kmeans_labels,n_itr_clusters = merge(mean_cluster_itr,std_cluster_itr,prob_cluster_itr,lambda_array,covar_D,clusters,kmeans_labels,n_itr_clusters,n_samples,n_dim_pca)
        mean_cluster, std_cluster, prob_cluster = cal_mean_var(final_train_data,clusters,n_itr_clusters)
        print "Merging Done"

    return clusters,kmeans_labels

if __name__ == '__main__':

    new_train_data, num_dim, num_samples = create_vectors()
    words = []	# List of words
    words = create_word_list()
    y_train = []	# List having the classes for each of the document instance
    y_train = create_classes()
    num_classes = len(np.unique(y_train))
    print "Data Read done"
    final_train_data, component_array = perform_pca(new_train_data, n_dim_pca)	#Data after pca
    print "PCA done"
    kmeans_5 = KMeans(n_clusters=n_clusters).fit(final_train_data)
    y_train_k = kmeans_5.labels_
    # K-means
    kmeans = KMeans(n_clusters=Kmeans_n_clusters).fit(final_train_data)
    [clusters, y_predict ] = create_clusters(kmeans, n_samples, n_dim_pca, Kmeans_n_clusters, y_train_k)
    [Map, FMWIndex] = permute_labels(y_train, y_predict, n_clusters, n_clusters)
    print FMWIndex
    # --------------------------------------------------------------------------------------
    
    # Agglomerative
    # agglomerative = AgglomerativeClustering(n_clusters=Kmeans_n_clusters,affinity='euclidean').fit(final_train_data)
    # agglomerative_labels = agglomerative.labels_
    # [clusters,y_predict] = create_clusters(agglomerative, n_samples, n_dim_pca,Kmeans_n_clusters)

    # agglomerative_5 = AgglomerativeClustering(n_clusters=n_clusters,affinity='euclidean').fit(final_train_data)
    # agglomerative_labels = agglomerative_5.labels_
    # [Map_A, FMWIndex_A] = permute_labels(agglomerative_labels, y_predict, n_clusters)

    # [Map, FMWIndex] = permute_labels(y_train, y_predict, n_clusters)
    # print FMWIndex, FMWIndex_A
    # --------------------------------------------------------------------------------------
    
    