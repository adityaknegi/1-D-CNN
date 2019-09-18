import numpy as np
from forward_conv_parallel import *
from feed_forward import *
def predict(X, y, parameters,layers_dims,conv_layer_dims):
    m = X.shape[1] # number of inputes
    predictions = np.zeros((1,m)) # class
    
    # Forward propagation
    tree = cov_parrale(parameters,conv_layer_dims,X)
    
    layer=len(conv_layer_dims)-1
    
    value=np.concatenate([tree['F'+str(layer)][x] for x in  range(parameters['no_filters'+str(layer)])],axis=0)
    for n in range(1,len(layers_dims)):
        if n<len(layers_dims)-1:
            A, cache = linear_activation_forward(value, parameters["W"+str(n)], parameters["b"+str(n)], "relu")
        else:
            A, cache = linear_activation_forward(value, parameters["W"+str(n)], parameters["b"+str(n)], "sigmoid")
        value=A

    for i in range(A.shape[1]):
        if A[:,i]>0.5:
            predictions[:,i]=1
        else:
            predictions[:,i]=0
    
        
    return  str(np.sum((predictions == y)/m))
