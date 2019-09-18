import numpy as np

from activation_function_forward_backward import *


def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW =  1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ,axis = 1,keepdims = True )
    dA_prev = np.dot(W.T,dZ)
    """
    Arguments:
    dZ -- Gradient with respect to ;
    cache -- tuple of values (A_prev, W, b) from forward propagation
    Returns:
    dA_prev -- Gradient of the cost with prev layer
    dW -- db  Gradient
    """
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    #Gradient with respect to prev and Gradient
    return dA_prev, dW, db
