import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import math
import scipy as scipy
import io
import csv
import operator
from sklearn.metrics.pairwise import cosine_similarity

def semanticusers():
	#Provide no.of semantic users k and no.of users whom semantic friends have to be generated. 
	k=5	#Here k={5,10,15, or 20}.
	users_under_consideration=5	#Set this value less if you are interested to see the result quickly. Otherwise this value should be<total number of users
	
	#Dimensionality reduction using PCA; reduced dimension = 2
	pca_m = PCA(n_components=2)
	with open("results/output.emb", 'r') as my_file:
		a = np.loadtxt(my_file,skiprows=1)
		B=a[:,0]
		A=scipy.delete(a,0,1)
		XX = pca_m.fit_transform(A)
		C=np.column_stack((B,XX))
		thefilesort = open('results/sorted.txt', 'w')
		thefile = open('results/similar.txt', 'w') 
		
		#Cosine similarity computation. This contain n*n comparisons. So, it will take time please be patient. Computed result will be available in similar.txt.
		print "Please wait. The computation of cosine similarity is under process... " 
		for row in C:
			for row2 in C:
				if row[0] != row2[0]:
					user1 = row[1:3].reshape(1,-1)
					user2 = row2[1:3].reshape(1,-1)
					similarity = cosine_similarity(user1,user2) 
					thefile.write('{} {} {}\n'.format(row[0], row2[0], similarity))                  
		thefile.close()
		
		# Sorting the user-user similarity values. Theresult will be stored in friend_dic dictionary as well as in the file sorted.txt for cross checking the results.
		reader = csv.reader(open("results/similar.txt"), delimiter=" ")
		friend_dic=dict()
		for line in sorted(reader, key=operator.itemgetter(0,2), reverse=False):
			user= int(float(line[0]))
			friend=int(float(line[1]))
			if user in friend_dic:
				if len(friend_dic[user])!=k:
					friend_dic[user].append(friend)
			else:
				if len(friend_dic)!=users_under_consideration:
					friend_dic[int(float(line[0]))]=[int(float(line[1]))]
			thefilesort.write('{}\t{}\t{}\n'.format(user, friend, line[2]))
		thefilesort.close()
	
	return friend_dic




   
