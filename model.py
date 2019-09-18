from paramerter_intialized import *
from forward_conv_parallel import *
from cost import *
from update import *
from  predict  import *
from feed_forward import *
from back_prop import *
from train_test_split import * 
from back_conv_pool_parallel import *
import matplotlib.pyplot as plt

def N_layer_model(X, Y, layers_dims,conv_layer_dims, learning_rate = 0.1, num_iterations = 500):
    """N layer convolution Neural network (layers_dims,conv_layer_dims)
    
    """
    grads={}
    costs = []                              # to keep track of the cost 
    train_test=[]
    X_train, X_test, y_train, y_test = split(X,Y)
    m = X_train.shape[1]                           # number of examples
    parameters ={}
    # Initialize parameters dictionary
    np.random.seed(8)
    # no of filter + size of filter
    intialize_parameters_n_layer_conv(conv_layer_dims,parameters)

    
    learning_rate_i=learning_rate
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID.

        #sore all cache
        # for all layers 
        tree = cov_parrale(parameters,conv_layer_dims,X_train)
    
                
        layer=len(conv_layer_dims)-1
        value=np.concatenate([tree['F'+str(layer)][x] for x in  range(parameters['no_filters'+str(layer)])],axis=0)
        cache_ANN=[]
        if i==0:
        	layers_dims=tuple([layers_dims[x] if x>0 else value.shape[0] for x in range(len(layers_dims))])
        	initialize_parameters_n_layes(layers_dims,parameters)

		
        for n in range(1,len(layers_dims)):
            if n<len(layers_dims)-1:
                A, cache = linear_activation_forward(value, parameters["W"+str(n)], parameters["b"+str(n)], "relu")
            else:
                A, cache = linear_activation_forward(value, parameters["W"+str(n)], parameters["b"+str(n)], "sigmoid")
            value=A
            cache_ANN.append(cache)
       
        # Compute cost 
        cost = compute_cost(A, y_train)
        
        # Initializing backward propagation
        dA = - (np.divide(y_train, A) - np.divide(1 - y_train, 1 - A))
        
        # Backward propagation. 
        
        # fully connected layer 
        for n in reversed(range(1,len(layers_dims))):
            if n==len(layers_dims)-1:
                dA , dW , db =linear_activation_backward(dA,cache_ANN[n-1],"sigmoid")
            else:
                dA , dW , db =linear_activation_backward(dA,cache_ANN[n-1],"relu")
            # Set grads
            grads["dW"+str(n)]=dW
            grads["db"+str(n)]=db
       
        # pooling and conv back prop 
        grads=back_prop_cov_layer(parameters,conv_layer_dims,tree,dA,grads)
        
        # Update parameters W1, b1, W2, b2
        
        parameters = update_parameters(parameters, grads, learning_rate,conv_layer_dims,layers_dims)
        

        step=200 # store after step iterations  
        if i % step == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
            train=predict(X_train, y_train,parameters,layers_dims,conv_layer_dims)
            test=predict(X_test, y_test,parameters,layers_dims,conv_layer_dims)
                                                             
    
            print('train {} test {} learning_rate {}'.format(train,test,learning_rate))
            #learning_rate =step_decay(i/100,learning_rate_i)
            train_test.append((train,test))
            
            #learning_rate =step_decay(i/100,learning_rate_i)
   
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations ({})'.format(step))
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    # return updated paramenters
    return parameters,tree,train_test
