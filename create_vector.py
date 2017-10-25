import sys, os
import numpy as np
from numpy import *

file_bbc_mtx = open("bbc/bbc.mtx", "r")

count_row = 0
train_data = np.zeros((2225, 9635))	# Sparse data matrix

for row in file_bbc_mtx:
	if count_row > 1:
		row  = row.split()
		#print row
		train_data[int(row[1])-1][int(row[0])-1] = float(row[2])

	count_row += 1

words = []	# List of words

file_bbc_terms = open("bbc/bbc.terms", "r")

for row in file_bbc_terms:
	row = row.split("\n")
	words.append(row[0])	# List of words

count_row = 0

classes = []	# List having the classes for each of the document instance

file_bbc_classes = open("bbc/bbc.classes", "r")

for row in file_bbc_classes:
	if count_row > 3:
		row = row.split()
		classes.append(int(row[1]))

	count_row += 1
