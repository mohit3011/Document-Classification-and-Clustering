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

x = []
y = []

def reset(temp_x, temp_y):
	x.append(temp_x)
	y.append(temp_y)
	temp_x_1 = []
	temp_y_1 = []
	return temp_x_1, temp_y_1

if __name__ == '__main__':


	file_object = open("graph_data.txt", "r")

	color_list = ["b", "g", "r", "c", "m", "y"]
	
	temp_x = []
	temp_y = []

	count = 0
	color_count = 0

	for line in file_object:
		
		line = line.split()
		#print line
		if count < 10:
			temp_x.append(float(line[0]))
			temp_y.append(float(line[1]))
			if count==9:
				temp_x, temp_y = reset(temp_x, temp_y)

		elif count>=10 and count < 20:
			temp_x.append(float(line[0]))
			temp_y.append(float(line[1]))

			if count==19:
				temp_x, temp_y = reset(temp_x, temp_y)

		elif count>=20 and count < 30:
			temp_x.append(float(line[0]))
			temp_y.append(float(line[1]))

			if count==29:
				temp_x, temp_y = reset(temp_x, temp_y)

		elif count>=30 and count < 40:
			temp_x.append(float(line[0]))
			temp_y.append(float(line[1]))

			if count==39:
				temp_x, temp_y = reset(temp_x, temp_y)

		elif count>=40 and count < 50:
			temp_x.append(float(line[0]))
			temp_y.append(float(line[1]))

			if count==49:
				temp_x, temp_y = reset(temp_x, temp_y)

		elif count>=50 and count < 60:
			temp_x.append(float(line[0]))
			temp_y.append(float(line[1]))

			if count==59:
				temp_x, temp_y = reset(temp_x, temp_y)


		count += 1

	#print x
	#print y
	plt.plot(y[0], x[0], color = color_list[0], linewidth=2.0, label="hs-lr-em")
	plt.plot(y[1], x[1], color = color_list[1], linewidth=2.0, label="he-lr")
	plt.plot(y[2], x[2], color = color_list[2], linewidth=2.0, label="he-fm")
	plt.plot(y[3], x[3], color = color_list[3], linewidth=2.0, label="he-fm-em")
	plt.plot(y[4], x[4], color = color_list[4], linewidth=2.0, label="km-fm")
	plt.plot(y[5], x[5], color = color_list[5], linewidth=2.0, label="km-lr")
	plt.legend(loc='upper left')

	plt.show()	