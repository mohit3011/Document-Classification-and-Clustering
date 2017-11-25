def calc_cosine(c1,c2,n_dim_pca):
    determinant_l_D = np.prod((c1['lambda']*c1['covar_D']) + (c2['lambda']*c2['covar_D']))
    lambda_merge = (c1['lambda']*c2['lambda'])/(determinant_l_D**(1.0/n_dim_pca))
    covar_D_merge = 1.0/((determinant_l_D**(1.0/n_dim_pca))*((c1['lambda']/(1.0*c1['covar_D'])) \
                                            + (c2['lambda']/(1.0*c2['covar_D']))))
    mean_merge = lambda_merge*covar_D_merge*(((1.0/(c1['lambda']*c1['covar_D']))*c1['mean']) \
                                            + (((1.0/(c2['lambda']*c2['covar_D']))*c2['mean'])))
    coefficient_term = (((c1['lambda']*2)**(n_dim_pca*0.25))*((c2['lambda']*2)**(n_dim_pca*0.25)))/(determinant_l_D**0.5)
    exponent_term = -0.5*(np.dot(c1['mean'],((1.0/(c1['lambda']*c1['covar_D']))*c1['mean'])) \
                            + np.dot(c2['mean'],(((1.0/(c2['lambda']*c2['covar_D']))*c2['mean']))) \
                            - np.dot(mean_merge,((1.0/(lambda_merge*covar_D_merge))*mean_merge)))
    cosine_score = coefficient_term * math.exp(exponent_term)
    return cosine_score

def merge(mean_cluster_itr,std_cluster_itr,prob_cluster_itr,clusters):
    curr_max = 0
    curr_max_pair = (-1,-1)
    for i in range(n_clusters):
        if len(clusters[i])== 0:
            continue
        cluster1 = defaultdict()
        cluster1['mean'] = mean_cluster_itr[i]
        cluster1['std'] = std_cluster_itr[i]
        cluster1['size_ng'] = int(prob_cluster_itr[i]*n_samples)

        ######## TO BE DONE #######
        cluster1['lambda'] = To_be_calcuated
        cluster1['covar_D'] = To_be_calcuated
        ###########################

        for j in range(i+1,n_clusters):
            if len(clusters[i])==0:
                continue
            cluster2 = defaultdict()
            cluster2['mean'] = mean_cluster_itr[j]
            cluster2['std'] = std_cluster_itr[j]
            cluster2['size_ng'] = int(prob_cluster_itr[j]*n_samples)
            ######## TO BE DONE #######
            cluster2['lambda'] = To_be_calcuated
            cluster2['covar_D'] = To_be_calcuated
            ###########################

            cosine_distance = calc_cosine(cluster1,cluster2)
            if(cosine_distance >= curr_max):
                curr_max = cosine_distance
                curr_max_pair = (i,j)

        #To merge i,j into i
        flag[j] = 1
        clusters[i] = np.append(clusters[i],clusters[j])
        clusters[j] = np.array([])
