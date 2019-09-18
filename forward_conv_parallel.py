# CORRECt check with tree layer 
from forward_1dconv import *
from forward_1dpool import *

def cov_parrale(parameters,conv_layer_dims,X):
    
    """multilayer convolution 
    
    Arguments:
    tree--- to store cache(for backprop operation) of convoltion and pooling and pooling value 
    total_filters --- filter to apply All pre channels
    tree['C'+str(l)]---store convolution cache 
    tree['P'+str(l)]---store pooling cache 
    tree['F'+str(l)]---store All pooling value 
    value--- previous layer value from All channels
    
    Return:
    
    tree--- store all cache of conv and pool values
    
    
    
    """
    # tree just for intiuation can store in paramters
    tree = {}
    tree['X']=X
    total_filters=1 # for start

    for l in range(len(conv_layer_dims)):
        # apply total filter on all channels 
        try:
            pre_filters=parameters['filters'+str(l-1)]
        except:
            pre_filters=1
        # apply no of filters
        total_filters=parameters['filters'+str(l)] 
        # store all cache of conv and pool values 
        tree['C'+str(l)]=[]
        tree['P'+str(l)]=[]
        tree['F'+str(l)]=[]
        for node in range(pre_filters):
            # for all sub filter after layer 
            try:
                # extract  node values 
                # pre layer all  values after Applyied conv and pooling 
                value=tree['F'+str(l-1)][node]


            except:
                value=tree['X'] # if first layer

            # filter one by one on one pre pool result 

            for n in range(total_filters):
                # same for all 
                loc=str(l)+str(n)
                C, cachec  = cov1D(value,parameters['C'+loc],activation='relu')    
                P, cachep  = pool_1D(C,parameters['P'+loc],method='max')
                # unique name by layer and new node 
                tree['C'+str(l)].append(cachec)
                tree['P'+str(l)].append(cachep)
                tree['F'+str(l)].append(P)
            # one child of each node 

            # cache save from layers 
    return tree

