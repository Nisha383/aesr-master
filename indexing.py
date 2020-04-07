#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

'''This function is used to handle variable length user IDs in the rating data. This will give unique indexing for each users and eliminate duplicate entriesin the dataset'''
def data_indexing(udata):
	uindex = {uid:idx for idx, uid in enumerate(udata['userid'].unique())}
	iindex = {iid:idx for idx, iid in enumerate(udata['itemid'].unique())}
	total_user = len(uindex.items())
	total_item = len(iindex.items())
	
	#Give the file name where the indexed rating data has to be stored
	f = open("results/indexed_data.txt", 'w')
	for line in udata.itertuples():
		f.write( str(uindex[line[1]]) +' '+str(iindex[line[2]]) +' '+str(line[3]) +'\n')
	f.close()
	
	indexed_data=pd.read_csv("results/indexed_data.txt", sep=' ', names=['userid', 'itemid', 'rating'], engine='python')
	
	return indexed_data
