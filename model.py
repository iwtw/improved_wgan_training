import os,sys
import functools
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.read
import tflib.ops.layernorm
import tflib.plot

H = 28
W = 24
DIM = 32
OUTPUT_DIM = 112*96*3
#OUTPUT_DIM = H*4*W*4*3
def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # Baseline (G: DCGAN, D: DCGAN)
    return ResnetGenerator, ResnetDiscriminator

    raise Exception('You must choose an architecture!')


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs):
    #if ('Discriminator' in name) and (MODE == 'wgan-gp'):
    if 'Discriminator' in name :
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True , mode="old"):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        #conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_1b       = functools.partial( SubpixelConv2D , input_dim=input_dim/2 , output_dim=output_dim/2 )
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if mode =="old":
        output = tf.nn.relu(output)
        output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
        output = tf.nn.relu(output)
        output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
        output = tf.nn.relu(output)
        output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
        output = Batchnorm(name+'.BN2', [0,2,3], output)
        output = shortcut + (0.3*output)
    else:
        dim = input_dim
        output = lib.ops.conv2d.Conv2D(name+".Conv2" , input_dim , dim , filter_size, output , stride = 1 )
        output = Batchnorm( name+".BN1" , [0,2,3] , output  )
        output = tf.nn.relu( output )
        output = lib.ops.conv2d.Conv2D(name+".Conv3" ,  dim , dim , filter_size , output ,stride =1  )
        output = Batchnorm( name+".BN2" , [0,2,3] , output)
        output = shortcut + output

    return  output

# ! Generators
def ResnetGenerator(n_samples, inputs, dim=DIM):

   # output = lib.ops.linear.Linear('Generator.Input', 128, 2*H*W*dim, inputs)
    #output = lib.ops.linear.Linear('Generator.Input' , 3 , 2*H*W*dim , inputs )
    output = lib.ops.conv2d.Conv2D('Generator.Conv1' , 3 , dim ,9, inputs , stride  = 1   )
    output = tf.nn.relu(output)
    shortcut1 = output


    for i in xrange(16):
        output = ResidualBlock('Generator.{}x{}_{}'.format(H,W,i), dim, dim, 3, output, resample=None , mode="new")

    output = lib.ops.conv2d.Conv2D('Generator.Out', dim, dim , 3 , output, he_init=False)
    output = Batchnorm( "Generator.BN3" , [0,2,3] , output)
    output = shortcut1 + output

    output = SubpixelConv2D(name = "Generator.Subpixel1" , input_dim = dim , output_dim = dim , filter_size = 3 , inputs = output , stride = 1 )
    output = tf.nn.relu(output)

    output = SubpixelConv2D(name = "Generator.Subpixel2" , input_dim = dim , output_dim = dim , filter_size = 3 , inputs = output , stride = 1 )
    output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D("Generator.Conv2" , dim , 3 , 9 , output , stride =1  )

 # somebody use tanh while others not ??
    output = tf.tanh(output )

    return tf.reshape(output, [-1, OUTPUT_DIM])
    return output


# ! Discriminators

def ResnetDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 112, 96])
    #output = inputs
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in xrange(2):
        output = ResidualBlock('Discriminator.112x96_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in xrange(2):
        output = ResidualBlock('Discriminator.56x48_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in xrange(2):
        output = ResidualBlock('Discriminator.28x24_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in xrange(2):
        output = ResidualBlock('Discriminator.14x12_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')

    output = tf.reshape(output, [-1, 7*6*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 7*6* 8*dim, 1, output)

    return tf.reshape(output/5. , [-1])
