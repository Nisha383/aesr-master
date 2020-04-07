import numpy as np
import pandas as pd
import os
from sklearn import model_selection
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys
from indexing import data_indexing
from nrl_using_node2vec import network_embedding
from graph_construction import user_user_graph

'''Recieves input data (both rating and explicit/implicit) and pre-process them'''
def input_data(dataset, social_info):
	'''Provide more datasets in "if" condition if you needed'''
	if dataset == "sample":
		#if it is the github dataset isert the github.txt file. If the dataset is another one input accordingly.
		udata = pd.read_csv("input/rating.txt", sep=' ', names=['userid', 'itemid', 'rating'], engine='python')
		udata=data_indexing(udata)
		rdata_train,rdata_test = load_rating_data(udata)
		print "\nLoading and indexing of the rating data completed....\n\n"
		
		if social_info=="explicit":
				#S.txt contains explicit social connections in the form of edgelist
				l1, l2  = np.genfromtxt("input/friends.txt", dtype='int').T
				sdata = load_semantic_data(l1,l2)
				print "Explicit social connections are loaded....\n"
		else:
			#go to graph construction procedure 
			print "Implicit social info found...Going to graph construction and network embedding.."
			user_user_graph()
			
			#Apply network embedding over the constructed edgelist. Please make sure that you have entered the edgelist generated from graph construction procedure.
			sdata=network_embedding("results/user_user.edgelist")
			print "Implicit social connections are extracted. The friend list is:"
			print sdata
			print "\n\n"

	else:
		print "Enter a valid dataset!!!"
	
	return rdata_train,rdata_test, sdata

'''Function to generate rating matrix from raw input data. Here, the data is split to training and testing set.	'''
def load_rating_data(rdata):
	
	udata = rdata
	udata.tail()	
	udata = udata[udata.rating >=0]
	map_uid_index = {uid:idx for idx, uid in enumerate(udata['userid'].unique())}
	map_iid_index = {iid:idx for idx, iid in enumerate(udata['itemid'].unique())}
	total_user = len(map_uid_index.items())
	total_item = len(map_iid_index.items())
	list_users = range(total_user)	
	
	normalize = lambda x: 5. if x == 5 else 4. if x == 4 else 3. if x == 3 else 2. if x == 2 else 1. if x == 1 else 0
	
	#training and test set splitting
	udata_train, udata_test = model_selection.train_test_split(udata, test_size=.4)
	train_matrix = np.zeros([total_user, total_item], dtype=float)
	test_matrix = np.zeros([total_user, total_item], dtype=float)
	full_matrix = np.zeros([total_user, total_item], dtype=float)
	data_map = [(train_matrix, udata_train), (test_matrix, udata_test), (full_matrix, udata)]
	
	#rating matrix generation
	for imatrix, idata in data_map:		
		for line in idata.itertuples():
		    imatrix[map_uid_index[line[1]],
		                 map_iid_index[line[2]]] = normalize(line[3])
	return train_matrix,test_matrix

'''If user provides explicit social connections, this function will generate a dictionary of {user: friends} values'''
def load_semantic_data(l1,l2):	 
	uniques = list(np.unique(l1))+list(np.unique(l2))    
	dic = {}
	for i in np.unique(uniques):
		a =list(l2[np.where(l1 == i)[0]])
		b =list(l1[np.where(l2 == i)[0]])
		c = list(np.unique(a+b))
		dic[i] = c		
	return dic 
	
