import numpy as np 
from activation_function_f  import *

def conv_backward(dH, cache,activation):
    
    '''
    The backward computation for a convolution function
    
    Arguments:

    dH -- gradient of the cost with respect to output of the conv layer (H)
    cache -- cache of values needed for the conv_backward()
    
    Returns:
    
    dX -- gradient of the cost with respect to input of the conv layer (X)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
    '''
    (linera_cache,activation_cache)= cache
    
    
    (X,W)=linera_cache
    
    # Retrieving information from the "cache"
    
    # Retrieving dimensions from X's shape
    n_H_prev,n_number = X.shape
    
    # Retrieving dimensions from W's shape
    f = W.shape[0]
    
    # Retrieving dimensions from dH's shape
    n_H = dH.shape[0]
    
    # Initializing dX, dW with the correct shapes
    dX = np.zeros(X.shape)
    dW = np.zeros(W.shape)
    m = dH.shape[1]
    
    # Looping over horizontal(h)  axis of the output
    
    # back prop to activation function
    if activation =='relu':
        dZ = relu_backward(dH, activation_cache)
    

    for h in range(f-1):
        dW[h,:]= np.sum(X[h:h+n_H,:]*dZ)/m
    # n_H_pre heigh of X (data)
    # n_H height of dH (pool results)
    # f  height of W (filter)
    m=0
    size_of_w=0
    for i in range(n_H_prev):
        if f>i:
            
            dX[i,:]=np.sum(dH[:i+1,:]*np.flip(W[:i+1],axis=0),axis=0)
        elif f<=i and i<n_H:
            m=m+1
            dX[i,:]=np.sum(dH[m:m+f,:]*np.flip(W,axis=0),axis=0)
        else:
            m=m+1
            size_of_w+=1
            dX[i,:]=np.sum(dH[m:,:]*np.flip(W[size_of_w:],axis=0),axis=0)
    return dX, dW
