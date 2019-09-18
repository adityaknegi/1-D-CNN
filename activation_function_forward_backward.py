import numpy as np
# forward activation

def sigmoid(Z):
    '''
    '''
    return 1/(1+np.exp(-Z)), Z

def relu(Z):
    '''
    return max value
    '''
    return np.maximum(0,Z), Z


# backword activation

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    # When z <= 0, you should set dz to 0 as well. 
    dZ[cache <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

