import sys
import pdb
import numpy as np
from create_vector import *

n_clusters = 5
n_samples = 2225

#kmeans = KMeans(n_clusters=n_clusters).fit(Final_train_data)

#kmeans_labels = kmeans.labels_

def cal_mean_var(final_train_data,clusters,kmeans):
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
    mean_cluster_array,,std_cluster_array,prob_cluster = cal_mean_var(final_train_data,clusters,kmeans)

#### Done Editing till here

clusters_itr = np.copy(clusters)
cluster = np.copy(kmeans_labels)
p_x = np.copy(prob_cluster)
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
	mean_vector_x = np.zeros((n_clusters,n_dim_pca))
	cov_vector_x = np.zeros((n_clusters,n_dim_pca))
	p_x = np.zeros(n_clusters)

	for j in range (n_clusters):
		clusters.append([])

	for j in range(2225):
		clusters[updates[j]].append(j)

	for j in range(n_clusters):
		for k in range(n_dim_pca):
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
