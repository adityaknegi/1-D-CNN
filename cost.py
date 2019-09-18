
import numpy as np

def compute_cost(AL, Y):
    m = Y.shape[1]

    
    #  prevent  NaN values in your cost function
    AL[np.where(AL==0)]=0.00001
    AL[np.where(AL==1)]=0.99999
    
    
    cost = -1/m * np.sum(np.multiply(np.log(AL), Y) + np.multiply(1 - Y, np.log(1 - AL)))


    cost = np.squeeze(cost)
    
    return cost