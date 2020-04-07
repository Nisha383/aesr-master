import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import matplotlib
matplotlib.use('Agg')
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import scipy as scipy

import math
import io
import csv
import operator
from topkuser import semanticusers

'''This is an edited version of original node2vec model. Here, all the parameters of the model is explicitly given for the ease of use.'''
def read_graph(input, directed):
	'''
	Reads the input network in networkx.
	'''
	G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
	for edge in G.edges():
		G[edge[0]][edge[1]]['weight'] = 1
	
	if not directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	output=open("results/output.emb",'w')
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=64, window=10, min_count=0, sg=1, workers=8)
	model.save_word2vec_format(output)
	 
	return 
	
	
def network_embedding(input):

	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	print "Entered into node2vec!"
	directed=False
	nx_G = read_graph(input,directed)
	G = node2vec.Graph(nx_G, directed, p=1, q=1)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(num_walks=10, walk_length=80)
	learn_embeddings(walks)
	print "Embedding completed.\n"
	
	#Top-k friend list creation   
  	friend_list=semanticusers()
  	
	return friend_list




