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

def calc_cosine(c1,c2,n_dim_pca):
    if (c1['lambda']*c2['lambda']==0):
        cosine_score = float("inf")
    else:
        Q = np.log(c1['lambda']*c1['covar_D'] + c2['lambda']*c2['covar_D'])
        logDet = np.sum(Q)
        Logdeterminant_l_D = logDet*(1.0/n_dim_pca)
        determinant_l_D = np.exp(Logdeterminant_l_D)
        lambda_merge = (c1['lambda']*c2['lambda'])/(determinant_l_D)
        covar_D_merge = 1.0/((determinant_l_D)*((c1['lambda']/(1.0*c1['covar_D'])) \
                                                + (c2['lambda']/(1.0*c2['covar_D']))))
        mean_merge = lambda_merge*covar_D_merge*(((1.0/(c1['lambda']*c1['covar_D']))*c1['mean']) \
                                                + (((1.0/(c2['lambda']*c2['covar_D']))*c2['mean'])))
        coefficient_term = math.log(((c1['lambda'])**(n_dim_pca*0.25))*((c2['lambda'])**(n_dim_pca*0.25)))-(logDet*0.5)
        exponent_term = -0.5*(np.dot(c1['mean'],((1.0/(c1['lambda']*c1['covar_D']))*c1['mean'])) \
                                + np.dot(c2['mean'],(((1.0/(c2['lambda']*c2['covar_D']))*c2['mean']))) \
                                - np.dot(mean_merge,((1.0/(lambda_merge*covar_D_merge))*mean_merge)))
        
        cosine_score = exponent_term + coefficient_term
    return cosine_score
        
def merge(mean_cluster_itr,std_cluster_itr,prob_cluster_itr,lambda_array,covar_D,clusters,kmeans_labels,n_itr_clusters,n_samples,n_dim_pca):
    curr_max = (-1)*float("inf")
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

            cosine_distance = calc_cosine(cluster1,cluster2,n_dim_pca)
            if(cosine_distance >= curr_max):
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