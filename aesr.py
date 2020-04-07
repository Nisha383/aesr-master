import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
import time
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import logging
import sys
from metrics import RMSE, MAE, create_roc_curve
from load_data import input_data
from lmfit import Model

'''This is the proposed Auoencoder based social recommender system'''
def aesr(train_data, test_data,s_matrix ):
	
	data_x_ = train_data  #training rating matrix as input
	S = s_matrix	#Friend list
	f=len(S)
	data_x = data_x_ + np.random.normal(0, 0.1, (len(data_x_), len(data_x_[0]))) #Adding noise to the data. data_x is the user rating vector
	data_x_i = data_x_.T
	data_xi = data_x_i + np.random.normal(0, 0.1, (len(data_x_i), len(data_x_i[0])))	#Item rating vector

	epoch = 50	#number of iterations. Set to low for small datasets. 
	
	#user encoding layer input size declaration
	input_dim = len(data_x[0])
	hidden_dim = 8	#=64 for other data sets
	batch_size = len(data_x[0])

	#item encoding layer input size declaration
	input_dim_i = len(data_xi[0])
	hidden_dim_i = hidden_dim #no. of hidden units are set same for both user and item encoding layer
	batch_size_i = len(data_xi)
	D_dim = 4	#Shared layer D ={64,126 or 256} for other data sets
	training_result = []

	sess = tf.Session()
	x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x') #user encoding layer input initialization
	x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_') #output initialization
	xi = tf.placeholder(dtype=tf.float32, shape=[None, input_dim_i], name='xi') #item encoding layer input initialization

	#user encoding layer weights and bias initialization
	enc_wh = tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32))
	enc_bh = tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32))
	enc_wd = tf.Variable(tf.truncated_normal([hidden_dim, D_dim], dtype=tf.float32))
	enc_bd = tf.Variable(tf.truncated_normal([D_dim], dtype=tf.float32))

	#item encoding layer weights and bias initialization
	enc_whi = tf.Variable(tf.truncated_normal([input_dim_i, hidden_dim_i], dtype=tf.float32))
	enc_bhi = tf.Variable(tf.truncated_normal([hidden_dim_i], dtype=tf.float32))
	enc_wdi = enc_wd
	enc_bdi = enc_bd

	#Computations in the encoding layers
	hu = tf.nn.relu(tf.matmul(x, enc_wh) + enc_bh, name='user_encoder')
	U = tf.nn.relu(tf.matmul(hu, enc_wd) + enc_bd, name='user_in_D')
	hi = tf.nn.relu(tf.matmul(xi, enc_whi) + enc_bhi, name='item_encoder')
	V1 = tf.nn.relu(tf.matmul(hi, enc_wdi) + enc_bdi, name='item_in_D')
	V = tf.transpose(V1)

	decoder = tf.matmul(U, V) #Decoder

	l1 = tf.placeholder(dtype=tf.float32, shape=[f], name='l1')
	
	#Objective functions
	loss = 0.5 * tf.sqrt(tf.reduce_mean(tf.square(x_ - decoder)))  #Loss function
	
	loss += ((5e-4)/2) * (tf.nn.l2_loss(enc_bh)+ tf.nn.l2_loss(enc_bd)+ tf.nn.l2_loss(enc_wh) + tf.nn.l2_loss(enc_wd) + tf.nn.l2_loss(enc_bhi)+ tf.nn.l2_loss(enc_bdi)+ tf.nn.l2_loss(enc_whi) + tf.nn.l2_loss(enc_wdi))	#Regularization function

	loss += ((1e-4)/2) * (tf.nn.l2_loss(l1))	#Semantic social regularization function
	

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(.01, global_step, 10000, 0.86, staircase=True)	#learning rate
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)		#Optimization using SGD	

	sess.run(tf.global_variables_initializer())
	
	print "AESR is working with the generated data...Please wait if you are working on larger datasets.."
	print "Epoch\tRMSE\t\tMAE"
	for i in range(epoch):
		u = sess.run(U, feed_dict={x:data_x, xi:data_xi, x_: data_x_ }) 	#Generating user features U
		ln = len(u)
		mean = []
		C = []
		
		#Computation of semantic social regularizer.
		for user in range(len(data_x)):
		   C = u[user]
		   c = []
		   for k,v in S.items():
				if k==user:	   
				   for j in v: 
				   	   
					  c.append(u[j])
					  a = tf.constant(np.array(C), dtype=tf.float32)
					  b = tf.constant(np.array(c), dtype=tf.float32)
				   l = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(a-b),1)))
				   ll = sess.run(l)
				   mean.append(ll)	
		   	   
		   
		sess.run(optimizer, feed_dict={x:data_x, xi:data_xi, x_: data_x_ , l1: mean})
		
		if (i+1) % 1 == 0:        
		    actual_loss = sess.run(loss, feed_dict={x:data_x, xi:data_xi, x_: data_x_,  l1: mean})
		    pred = sess.run(decoder, feed_dict={x:data_x, xi:data_xi, x_: data_x_, l1:mean})
		    
		    #Evaluation measures
		    rmse = RMSE(test_data, pred)
		    mae = MAE(test_data, pred)
		    auc = create_roc_curve(test_data, pred, 1)
		    print i,"\t", np.round(rmse, 7),"\t", np.round(mae, 7)
	    
	pred = sess.run(decoder, feed_dict={x:data_x, xi:data_xi, x_: data_x_, l1:mean})
	print "\nTraining completed!!!!!!\n\n"
	
	#Finding the order of item indices the user may like in future. This will be helpfull for top-N recommendation. Note that the row index of the resulting matrix specifies the corresponding user index. The item idex is same as the entries of the matrix.
	order_pred = np.argsort(-pred)
		
	return pred, order_pred
 

def process(args):
	
	rating_data = args.ratingdataset
	friend_data = args.friendlist
	
	r_data_train,r_test, s_data = input_data(rating_data, friend_data)
	rating_predicted, op = aesr(r_data_train, r_test, s_data)
	
	print "Predicted Rating"
	print rating_predicted
	print "\nOrder of Prediction"
	print op
	
	return


def main():
  parser = ArgumentParser('Work2',
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
                          
  parser.add_argument('--ratingdataset', nargs='?', required=True, help='Input user-item rating file')
  parser.add_argument('--friendlist', nargs='?', required=True, help='explicit - if friend list is directly given, implicit - if friend list has to be generated using network embedding')

  args = parser.parse_args()

  process(args)

if __name__ == '__main__':
	sys.exit(main())	
	
