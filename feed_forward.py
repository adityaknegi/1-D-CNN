import numpy as np
from activation_function_f import *

def linear_forward(A, W, b):

    """
    liner forward 
    
    Arguments:
    
    A : input value 
    W : weights 
    b : bias 
    
    Returns:
    
    Z : 
    cache: value needed for  backpropagtion  

    """

    Z = np.dot(W,A)+ b 
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
	
    #liner forward and then passing through activation function 
    #Arguments:
    #A_prev --- value from previous layer 
    #W --- weights 
    #b --- bias
    #activation --- for non-linearity
    
    #Return 
    
    #Z --- Output value  after passing Neuron
    #A --- After passing by activation Function
    #cache --- Value Needed for  Backpropogtion

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache

