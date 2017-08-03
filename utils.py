import tensorflow as tf
import numpy as np
import time as time
from tensorflow.contrib import layers
def conv2d( inputs , outputs_dim , kernel_size ,   stride , padding = "SAME" , he_init = False , activation_fn = None , seed = None , regularization_scale = 0.0): 
    C = inputs.get_shape()[-1].value
    fan_in = C * kernel_size**2
#    print(fan_in)
    if he_init:
        var = 2.0/fan_in
    else :
        var = 1.0/fan_in
    # var = (b - a)**2 / 12 , b==-a(zero mean)
    upper_bound = np.sqrt( 12.0*var ) * 0.5 
    weights_initializer = tf.random_uniform_initializer( -upper_bound , upper_bound , seed = seed , dtype = tf.float32 )
    weights_regularizer = layers.l2_regularizer( scale = regularization_scale )
    return layers.conv2d( inputs = inputs , num_outputs = outputs_dim , kernel_size = kernel_size , stride =  stride, padding = "SAME"  , activation_fn = activation_fn , weights_initializer = weights_initializer  , weights_regularizer = weights_regularizer )

def fully_connected( inputs , outputs_dim , he_init = False , activation_fn = None , seed = None , regularization_scale = 0.0 ):
    x = layers.flatten( inputs )
    C = x.get_shape()[-1].value
    if he_init:
        var = 2.0/C
    else:
        var = 1.0/C
    # var = (b - a)**2 / 12 , b==-a(zero mean)
    upper_bound = np.sqrt( 12.0 * var ) *0.5
    weights_initializer = tf.random_uniform_initializer( -upper_bound , upper_bound , seed = seed , dtype = tf.float32 )
    #seed += 1
    weights_regularizer = layers.l2_regularizer( scale = regularization_scale )
    return layers.fully_connected( x , outputs_dim , weights_initializer =  weights_initializer , activation_fn = activation_fn  , weights_regularizer = weights_regularizer )
def basic_block(inputs , outputs_dim , kernel_size , stride  , he_init = False , activation_fn = None , regularization_scale = 0.0 ):
    f = conv2d( inputs , outputs_dim = outputs_dim , kernel_size = kernel_size , stride = stride , he_init = he_init , activation_fn = activation_fn  , regularization_scale = regularization_scale )
    f = layers.batch_norm( f , activation_fn = None )
    f = conv2d( f , outputs_dim = outputs_dim , kernel_size = kernel_size , stride = 1 , he_init = False , activation_fn = None , regularization_scale = regularization_scale )
    f = layers.batch_norm( f , activation_fn = None )

    if outputs_dim == f.shape[-1].value and stride == 1 :
        shortcut = inputs
    else:
        shortcut = conv2d( inputs , outputs_dim = outputs_dim , kernel_size = 1 , stride = stride , he_init = False , activation_fn = None , regularization_scale = regularization_scale ) 
    return activation_fn( shortcut + f )
def bottleneck_block(inputs , outputs_dim , intermediate_dim , kernel_size , stride , he_init = False , activation_fn = None , regularization_scale = 0.0 ):
    f = conv2d( inputs , outputs_dim = intermediate_dim , kernel_size = 1 , stride = stride , he_init = he_init , activation_fn = activation_fn  , regularization_scale = regularization_scale )
    f = conv2d( f , outputs_dim = intermediate_dim , kernel_size = kernel_size , stride = 1 , he_init = he_init , activation_fn = activation_fn , regularization_scale = regularization_scale )
    f = conv2d( f , outputs_dim = outputs_dim , kernel_size = 1 , stride = 1 , he_init = False , activation_fn = None , regularization_scale = regularization_scale )


    if outputs_dim == f.shape[-1].value and stride == 1 :
        shortcut = inputs
    else:
        shortcut = conv2d( inputs , outputs_dim = outputs_dim , kernel_size = 1 , stride = stride , he_init = False , activation_fn = None , regularization_scale = regularization_scale ) 
    return activation_fn( shortcut + f )
