import numpy as np 

def pool_1D(X,size=2,method='max'):
    """Pooling operation to reduce size 
    
    Arguments:
    
    X --- input Image
    size : pool filter size
    index_matrix--- 1 if max value in Matrix else 0
    
    Returns 
    
    new_matrix --- After Max pooling Results.
    cache  ---  value needed for  backpropagation   
    """
    X_row,x_number =X.shape
    # new row and according to filter size
    new_row =X_row//size 
    
    new_matrix = np.zeros((new_row,x_number))
    index_matrix=np.zeros((new_row,x_number))
    
    for start_row in  range(new_row):
            if method=='max':
                value=np.max(X[start_row*size:start_row*size+size,:],axis=0)
                i=(1+np.argmax(X[start_row*size:start_row*size+size,:],axis=0))*(start_row*size+1)
            else:
                value=np.mean(X[start_row*size:start_row*size+size,:],axis=0)
                i=(1+np.argmax(X[start_row*size:start_row*size+size,:],axis=0))*(start_row*size+1)
                
            new_matrix[start_row]=value
            index_matrix[start_row]=i-1    
            
    cache=(X,method,size,index_matrix.astype(int))
    
    return new_matrix ,cache

