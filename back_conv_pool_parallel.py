# back prop conv layers 
import numpy as np
from activation_function_f import *
from backward_1dpool import *
from backward_1dconv import *



def back_prop_cov_layer(parameters,conv_layer_dims,tree,value,grads):
	#Backprop for pooling and conv layers 

	#Arguments:
	#tree -- cache of values needed for  back_propopogation of layers 
	#total_no_filters -- All output filters in layer
	#filters -- All filters allpied for layer example 2 filter of size 3

	#returns: 
	#grads -- dW1,dC1 .. all gradient  for convolution layer
   	
 

    back_value=value
    for layer in reversed(range(len(conv_layer_dims))):
        
        filters=parameters['filters'+str(layer)]
        total_no_filters=parameters['no_filters'+str(layer)]
        
        sub_filter =total_no_filters//filters

        cache_conv=tree['C'+str(layer)]
        cache_pool=tree['P'+str(layer)]
        cache_pool_value=tree['F'+str(layer)]
        #  total_no_filters in layer l
        n=0
        value=back_value
        back_value=0
        end=0
        for node in range(sub_filter):
            for f in range(filters):
                # start location 
                top=0+end
                # end location 
                end=cache_pool_value[n].shape[0]+top
                dAp, method = pool_backward(value[top:end,:], cache_pool[n])
                dAc, dC = conv_backward(dAp, cache_conv[n],'relu')
                n=n+1
                # Set grads
                try:
                    grads['dC'+str(layer)+str(f)]+=dC
                except:
                    grads['dC'+str(layer)+str(f)]=dC
                # store value for backprop
                try:
                    back_value=np.concatenate((back_value,dAc),axis=0)
                except:
                    back_value=dAc
                
                
                
    return grads
