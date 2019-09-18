
def update_parameters(parameters, grads, learning_rate,conv_layer_dims,layers_dims):
    
    N = len(parameters) // 2 # number of layers in the neural network

    # Update prameters W1,W2,b1,b2
    for n in range(len(layers_dims)-1):
        parameters["W" + str(n+1)] = parameters["W" + str(n+1)] - learning_rate * grads["dW" + str(n+1)]
        parameters["b" + str(n+1)] = parameters["b" + str(n+1)] - learning_rate * grads["db" + str(n+1)]
    # covolution neural network 
    for layer in range(len(conv_layer_dims)):
        for  n in range(parameters['filters'+str(layer)]):
            parameters["C"+str(layer)+str(n)] = parameters["C"+str(layer)+str(n)] - learning_rate * grads["dC"+str(layer)+str(n)]
    #print(grads["dC0"] )
        
    
    return parameters
