import numpy as np

def initialize_parameters_n_layes(layer_dims,parameters):
    N = len(layer_dims)            # number of layers in the network
    
    # * np.sqrt(1/layer_dims[n]) for gradient vanishing and exploding 
    
    # intialized parameters layer by layer
    for n in range(1, N):
        parameters['W' + str(n)] = np.random.randn(layer_dims[n], layer_dims[n-1]) * np.sqrt(1/layer_dims[n])
        parameters['b' + str(n)] = np.zeros((layer_dims[n], 1))*np.sqrt(1/layer_dims[n])
        
    return parameters

def intialize_parameters_n_layer_conv(conv_layer_dims,parameters):
    pre=1
    for l in range(len(conv_layer_dims)):
        
        # filters no used in eqach layer and size
        
        (parameters['filters'+str(l)],parameters["f_size"+str(l)])=conv_layer_dims[l]
        pre=parameters["filters"+str(l)]*pre
        
        # total filter in each layer 2X2X5=20
        parameters['no_filters'+str(l)]=pre
        
        for n in range(parameters['filters'+str(l)]):
            # layer the number of filter
            np.random.seed(8)
            parameters['C'+str(l)+str(n)]=np.random.uniform(0,1,
                                                     parameters["f_size"+\
                                                                str(l)]).reshape(parameters["f_size"+str(l)],
                                                                                          1)*np.sqrt(1/parameters["f_size"+str(l)])
            parameters['P'+str(l)+str(n)]=2
    
    return 
