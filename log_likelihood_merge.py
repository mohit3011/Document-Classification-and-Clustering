from newGMM import *

n_samples = 2225 

def log_likelihood(c1,c2):
	global n_dim_pca
	mean_new = (1.0*(c1['size_ng']*c1['mean']+c2['size_ng']*c2['mean']))/(c1['size_ng']+c2['size_ng'])
	w_new = c1['std']+c2['std']+c1['size_ng']*(mean_new-c1['mean'])*(np.transpose((mean_new-c1['mean'])))+c2['size_ng']*(mean_new-c2['mean'])*(np.transpose((mean_new-c2['mean'])))
	lambda_new = (np.prod(diag(w_new))**(1.0/n_dim_pca))/(c1['size_ng']+c2['size_ng'])
	likelihood = n_dim_pca*(c1['size_ng']+c2['size_ng'])*np.log(lambda_new)-n_dim_pca*c1['size_ng']*np.log(c1['lambda'])-n_dim_pca*c2['size_ng']*np.log(c2['lambda'])
	return likelihood

def merge_log(mean_cluster_itr,std_cluster_itr,prob_cluster_itr,lambda_array,covar_D,clusters,kmeans_labels,n_itr_clusters):
	curr_max = 0
	curr_max_pair = (-1,-1)
	for i in range(n_itr_clusters):
	    #if len(clusters[i])== 0:
	    #    continue
	    cluster1 = defaultdict()
	    cluster1['mean'] = mean_cluster_itr[i]
	    cluster1['std'] = std_cluster_itr[i]
	    cluster1['size_ng'] = int(prob_cluster_itr[i]*n_samples)

	    ######## TO BE DONE #######
	    cluster1['lambda'] = lambda_array[i]
	    cluster1['covar_D'] = covar_D[i]
        ###########################

        for j in range(i+1,n_itr_clusters):
			#if len(clusters[i])==0:
			#    continue
			cluster2 = defaultdict()
			cluster2['mean'] = mean_cluster_itr[j]
			cluster2['std'] = std_cluster_itr[j]
			cluster2['size_ng'] = int(prob_cluster_itr[j]*n_samples)
			######## TO BE DONE #######
			cluster2['lambda'] = lambda_array[j]
			cluster2['covar_D'] = covar_D[j]
			###########################

			#cosine_distance = calc_cosine(cluster1,cluster2)
			#if(cosine_distance >= curr_max):
			#    curr_max = cosine_distance
			#    curr_max_pair = (i,j)

			likelihood = log_likelihood(cluster1,cluster2)
			if(likelihood <= curr_max):
				curr_max = likelihood
				curr_max_pair = (i,j)


        #To merge i,j into i
	i = curr_max_pair[0]
	j = curr_max_pair[1]
	clusters[i] = np.append(clusters[i],clusters[j])
	clusters = np.append(clusters[:j],clusters[j+1:])
	n_itr_clusters -= 1
	kmeans_labels[kmeans_labels==j]=i
	kmeans_labels[kmeans_labels>j]-=1

	return clusters,kmeans_labels,n_itr_clusters
