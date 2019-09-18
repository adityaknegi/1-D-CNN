import numpy as np 
from activation_function_f import relu
def cov1D(X,W,activation='relu'):

    """convolution operation on dataset 
    
    Arguments:
    
    X -- output of the previous layer, numpy array of shape (n_H_prev, n_W_prev) 
    W -- Weights, numpy array of size (f) assuming number of filters = 1
    
    Returns:
    
    new_matrix --- is output convolution
    cache --- cache needed for backpropagation
    """
    W_row= W.shape[0]
    X_row, x_number=X.shape
    # new row and col
    new_row =X_row -W_row+1
    new_matrix = np.zeros((new_row,x_number))

    for start_row in  range(new_row):
        new_matrix[start_row,:]=np.sum(X[start_row:start_row+W_row,:]*W.reshape(-1,1),axis=0)
    # if odd output then add zero (padding) to make even  

    if new_matrix.shape[0]%2==1:
        new_matrix=np.concatenate((new_matrix,np.zeros((1,new_matrix.shape[1]))), axis=0)
    
    
    linera_cache=(X,W)
    if activation=='relu':
        A, activation_cache = relu(new_matrix)

        
    cache =(linera_cache,activation_cache)
    
    return A,cache
