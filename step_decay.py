import numpy as np
def step_decay(steps,learning_rate):
	    
    #Learning rate decay function 
    
    #Arguments:

    #intial_lrte --- starting lerning rate
    #drop --- how much drop should be
    #epochs_drop --- after how many epoch decy should happen 
    
    #Returns:
    
    #learning_rate --- updated learning rate
    
    initial_lrate = learning_rate
    drop = 0.2
    epochs_drop = 2
    learning_rate = initial_lrate * np.power(drop,np.floor((1+steps)/epochs_drop))
    return learning_rate