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
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

n_dim_pca = 100



if __name__ == '__main__':

	weight_type_list = ["Entropy", "Identity"]
	transform_type_list = ["Log", "Identity", "Sqrt"]
	color_list = ["b", "g", "r", "c", "m", "y"]
	line_type_list = [ ]
	words = []	# List of words
	words = create_word_list()
	y = []	# List having the classes for each of the document instance
	y = create_classes()
	num_classes = len(np.unique(y))
	color_count = 0
	for weight_type in weight_type_list:
		for transform_type in transform_type_list:
			new_train_data, num_dim, num_samples = create_vectors(weight_type, transform_type)
			dim_list = []
			accuracy_list = []
			print "weight_type : ", weight_type, " transform_type : ", transform_type
			temp_dim = 20
			while temp_dim < 100:		
				final_train_data, component_array = perform_pca(new_train_data, temp_dim)	#Data after pca
				x_train, x_test, y_train, y_test = train_test_split(final_train_data, y, test_size=0.3, random_state=42)
				print "done splitting"
				neigh = KNeighborsClassifier(n_neighbors=5)
				neigh.fit(x_train, y_train)
				print "done knn fit"
				y_predict = neigh.predict(x_test)
				count_correct = 0
				print "done predict"
				for i in range(len(y_predict)):
					if y_predict[i]==y_test[i]:
						count_correct += 1
				accuracy = float(count_correct)/float(len(y_predict))
				dim_list.append(temp_dim)
				accuracy_list.append(accuracy)
				print "temp_dim : ", temp_dim, " accuracy : ", accuracy
				temp_dim += 20

			plt.plot(dim_list, accuracy_list, color = color_list[color_count], linewidth=2.0)
			color_count += 1
	plt.show()

