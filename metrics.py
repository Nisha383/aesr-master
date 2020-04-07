import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
from math import sqrt


def RMSE(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))


def MAE(y_actual, y_predicted):
	return mean_absolute_error(y_actual, y_predicted)


def create_roc_curve(labels, scores, poslabel ):
	m, n = labels.shape 
	y_true = []
	y_pred = []
	for i in range(m):
		for j in range(n):
			if labels[i][j]>=3:	
				y_true.append(1)
			else:
				y_true.append(0)
	for i in range(m):
		for j in range(n):
			if scores[i][j]>=3:
				y_pred.append(1)
			else:
				y_pred.append(0)			  			
	fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=poslabel)
	roc_auc = auc(fpr, tpr)
	
	return roc_auc
