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

def log_likelihood(c1,c2,n_dim_pca):
	c1['std']=c1['std'] * c1['std']
	c2['std']=c2['std'] * c2['std']
	mean_new = (1.0*(c1['size_ng']*c1['mean']+c2['size_ng']*c2['mean']))/(c1['size_ng']+c2['size_ng'])
	w_new = c1['std']+c2['std']	\
				+c1['size_ng']*(mean_new-c1['mean'])*(mean_new-c1['mean'])	\
				+c2['size_ng']*(mean_new-c2['mean'])*(mean_new-c2['mean'])
	Q=np.log(w_new)
	logDet=np.sum(Q)
	Log_determinant_l_D=float(logDet)/float(n_dim_pca)
	Log_lambda_new = Log_determinant_l_D - math.log(c1['size_ng']+c2['size_ng'])

	likelihood = n_dim_pca*((c1['size_ng'] + c2['size_ng'])*Log_lambda_new	\
					-c1['size_ng']*np.log(c1['lambda'])	\
					-c2['size_ng']*np.log(c2['lambda']))
	return likelihood
        
def merge_log(mean_cluster_itr,std_cluster_itr,prob_cluster_itr,lambda_array,covar_D,clusters,kmeans_labels,n_itr_clusters,n_samples,n_dim_pca):
    curr_max = float("inf")
    curr_max_pair = (-1,-1)
    for i in range(n_itr_clusters):
        cluster1 = defaultdict()
        cluster1['mean'] = mean_cluster_itr[i]
        cluster1['std'] = std_cluster_itr[i]
        cluster1['size_ng'] = int(prob_cluster_itr[i]*n_samples)
        cluster1['lambda'] = lambda_array[i]
        cluster1['covar_D'] = covar_D[i]

        for j in range(i+1,n_itr_clusters):
            cluster2 = defaultdict()
            cluster2['mean'] = mean_cluster_itr[j]
            cluster2['std'] = std_cluster_itr[j]
            cluster2['size_ng'] = int(prob_cluster_itr[j]*n_samples)
            cluster2['lambda'] = lambda_array[j]
            cluster2['covar_D'] = covar_D[j]

            cosine_distance = log_likelihood(cluster1,cluster2,n_dim_pca)
            if(cosine_distance <= curr_max):
                curr_max = cosine_distance
                curr_max_pair = (i,j)

    #To merge i,j into i
    i = curr_max_pair[0]
    j = curr_max_pair[1]
    clusters[i] = clusters[i]+clusters[j]
    if (j==n_itr_clusters-1):
        clusters = clusters[:j]
    else:
        clusters.pop(j)
    n_itr_clusters -= 1
    kmeans_labels[kmeans_labels==j]=i
    kmeans_labels[kmeans_labels>j]-=1
    return clusters,kmeans_labels,n_itr_clusters