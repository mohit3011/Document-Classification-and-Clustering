import sys
import pdb
import numpy as np
from create_vector import *

n_clusters = 5
n_samples = 2225


def cal_mean_var(final_train_data, clusters):
    mean_cluster_array = []
    std_cluster_array = []
    prob_cluster = []
    for i in range(n_clusters):
        data_indices = np.asarray(clusters[i])
        data_cluster = final_train_data[data_indices]
        mean_cluster = data_cluster.mean(axis=0)
        std_cluster = data_cluster.std(axis=0)
        mean_cluster_array.append(mean_cluster)
        std_cluster_array.append(std_cluster)
        prob_cluster.append((1.*np.size(clusters[i]))/n_samples)

    return mean_cluster_array,std_cluster_array,np.asarray(prob_cluster)



def create_clusters(kmeans):
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


	for i in range(10):
		mean_cluster_copy = np.copy(mean_cluster_itr)
		std_cluster_copy = np.copy(std_cluster_itr)
		updates_labels = np.copy(kmeans_labels_itr)
		prob_cluster_copy = np.copy(prob_cluster_itr)

		for k in range(n_samples):
			
			find_cluster = 0
			prob_cluster_max = float("inf")
			
			f = final_train_data[k]
			p_copy = np.copy(prob_cluster)
			
			for j in range(n_clusters):
				if np.size(clusters[j])!=1:
					mu = mean_cluster_copy[j]
					sig = std_cluster_copy[j]
					
					p_copy[j]=-1*((np.log(p_copy[j]))+(np.sum(norm.logpdf(f,mu,sig))))
					
					if(p_copy[j] < prob_cluster_max):
						prob_cluster_max = p_copy[j]		#Problem maybe
						find_cluster = j

			updates_labels[k] = find_cluster
		
		clusters = []

		for j in range (n_clusters):
			clusters.append([])

		for j in range(2225):
			clusters[updates_labels[j]].append(j)

		mean_cluster_itr, std_cluster_itr, prob_cluster_itr = cal_mean_var(final_train_data,clusters)
		
		kmeans_labels_itr = updates_labels


rand.seed(42)


if __name__ == '__main__':
	new_train_data, num_dim, num_samples = create_vectors()
	words = []	# List of words
	words = create_word_list()
	y_train = []	# List having the classes for each of the document instance
	y_train = create_classes()
	num_classes = len(np.unique(y_train))
	final_train_data, component_array = perform_pca(new_train_data, num_samples) #data after pca
    kmeans = KMeans(n_clusters=n_clusters).fit(Final_train_data) #k-means clustering
    kmeans_labels = kmeans.labels_
