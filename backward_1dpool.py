import numpy as np 
def pool_backward(dH, cache):
    """The backward computation for a pool for max pooling 
    Arguments:

    dH -- gradient of the cost with respect to output of the pool layer (H)
    cache -- cache of values needed for the pool_backward()
    
    Returns:
    
    dX -- gradient of the cost with respect to input of the conv layer (X)
    method -- method used max or avg
        
    """    
    # Retrieving information from the "cache"

    (X,method,size,index_matrix)= cache
    
    # Retrieving dimensions from X's shape
    n_H_prev,n_number = X.shape

    
    # Retrieving dimensions from dH's shape
    n_H = dH.shape[0]
    m = dH.shape[1]
    
    
    # Initializing dX
    dX = np.zeros((dH.shape[0]*size,n_number))

    #There is no gradient with respect to non maximum values so only max is locally linear with slope 1
    
    for i in range(n_H):
        dX[index_matrix[0],np.arange(0,index_matrix.shape[1],1)]=dH[i]
    
    
    return dX, method 