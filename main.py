
from data import *
from model import *
import numpy as np 



# layer dimention
'''
conv layer first parameter is no of filter then size of filter. (3,3) 3 filters, with size of 3

'''

X,y=wbc()
for X,y  in (wbc(),ionosphere(),liver(),LSVT_voice_rehabilitation()):
	conv_layer_dims=[(3,3),(2,1)]
	layers_dims = (0,100, 1)
	
	'''
	preprocessing data 
	'''
	parameters,tree ,train_test= N_layer_model(X,y,
	                                layers_dims = layers_dims,
	                                conv_layer_dims=conv_layer_dims,
	                                learning_rate=0.05,
	                                num_iterations = 2001)

