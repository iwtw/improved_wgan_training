import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.read
import tflib.ops.layernorm
import tflib.plot
import os
import sys
import vgg19

TEST_SPEED = False
TENSORFLOW_READ = False

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("output"):
    os.mkdir("output")

#LEARNING_RATE=tf.Variable( 5e-5 )
LOG_STEP=1000
P =float( sys.argv[2] )
DIM = 32 
N_GPUS = 2 
BATCH_SIZE = N_GPUS * 64 
NUM_EPOCHS = 25
LAMBDA = 10 
OUTPUT_DIM = 112*96*3 
DATA_TRAIN = "data.train"
DATA_VAL = "data.val"
data_train = open(DATA_TRAIN ).read().split('\n')
data_train.pop(len(data_train) -1 )
data_train = np.array( data_train )
data_val = open(DATA_VAL).read().split('\n')
data_val.pop(len(data_val)-1)
data_val = np.array( data_val )

H = 28
W = 24
NAME = ""
if ( len(sys.argv) > 1  ):
    NAME  = sys.argv[1]
    OUTPUT_PATH = "output/" + NAME 
    CHECKPOINT_PATH = "checkpoint/" + NAME
else:
    OUTPUT_PATH = "output/default"
    CHECKPOINT_PATH= "checkpoint/default"
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)

DATA_SIZE = len(data_train)
EPOCH_SIZE = DATA_SIZE / BATCH_SIZE 
lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """
    # Baseline (G: DCGAN, D: DCGAN)
    return ResnetGenerator, ResnetDiscriminator

    raise Exception('You must choose an architecture!')

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

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


Generator = ResnetGenerator



DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False )
config.gpu_options.allow_growth=True
with tf.Session(config=config) as session:
    

    minibatch = tf.placeholder( tf.uint8 , shape =(BATCH_SIZE , H*4 , W * 4 , 3 )  )


    gen_costs   , fake_datas , real_datas , x_bicubics = [] ,  [] , [] , []
    split_minibatch  = tf.split( minibatch , len(DEVICES) , axis = 0  )
    for device_index , device in enumerate(DEVICES):
        with tf.device(device):
            minibatch_idx = split_minibatch[device_index]
            minibatch_lr = tf.image.resize_bicubic( minibatch_idx , [ H,W ]  )
            minibatch_bicubic = tf.clip_by_value ( tf.image.resize_bicubic ( minibatch_lr , [H*4,W*4] )  , 0 , 255 )
           
            
            x_lr = tf.transpose( minibatch_lr , [0,3,1,2] )
            x = tf.transpose( minibatch_idx , [0,3,1,2])
            x_bicubic = tf.transpose( minibatch_bicubic , [0,3,1,2] )


            real_data = tf.reshape(2*((tf.cast( x , tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE/len(DEVICES) , inputs = x_lr )

            
            vgg_inputs = tf.transpose ( tf.concat( tf.reshape( [real_data , fake_data] , [-1,3,112,96] ), axis = 0 ) , [0,2,3,1 ] )
            vgg=vgg19.Vgg19()
            vgg.build( vgg_inputs )
            fmap=tf.split(vgg.conv2_2,2)
            vgg_loss = tf.losses.mean_squared_error( fmap[0] , fmap[1] )

            gen_content_cost = tf.losses.mean_squared_error( real_data , fake_data )
            gen_cost = (1 -P ) * gen_content_cost + P * vgg_loss 

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES),1], 
                minval=0.,
                maxval=1.
            )

            gen_costs.append(gen_cost)
            real_datas.append(real_data) 
            x_bicubics.append(x_bicubic)
            fake_datas.append(fake_data)

    gen_cost = tf.add_n(gen_costs)/len(DEVICES)  
    real_data = tf.concat( real_datas , axis = 0 )
    fake_data = tf.concat( fake_datas , axis = 0 ) 
    x_bicubic = tf.concat( x_bicubics , axis = 0 )



    global_step = tf.Variable( initial_value = 0 , dtype = tf.int32 , trainable = 0 ,name = 'global_step')
    #boundaries = [ epoch_size * 6 , epoch_size * 11 , epoch_size * 16 ]
    boundaries = [ 6 * EPOCH_SIZE ,  11 * EPOCH_SIZE , 16 * EPOCH_SIZE , 21 * EPOCH_SIZE]
    lrs = [ 1e-3 , 1e-4 , 5e-5 , 1e-5 , 1e-6 ]
    lr = tf.train.piecewise_constant( global_step , boundaries , lrs  )
    gen_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(gen_cost,
                                      var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), colocate_gradients_with_ops=True , global_step = global_step)

    
    # For generating samples
    def generate_image(it):

        _get_batch = lib.read.get_batch( data_train , BATCH_SIZE )
        real_samples , samples , bicubic  = session.run( [real_data , fake_data  , x_bicubic ] , feed_dict = { minibatch:_get_batch }) 


        samples = ((samples+1.)*(255.99/2)).astype('uint8')
        samples = samples.reshape((BATCH_SIZE, 3, H*4, W*4))
        real_samples = ((real_samples+1.)*(255.99/2)).astype('uint8')
        real_samples = real_samples.reshape((BATCH_SIZE, 3, H*4, W*4))
        bicubic = bicubic.astype('uint8')
        
        with tf.device('/cpu:0'):
            lib.save_images.save_images( samples , OUTPUT_PATH+'/samples_{}_gen.png'.format(it))
            lib.save_images.save_images( real_samples , OUTPUT_PATH+'/samples_{}_real.png'.format(it))
            lib.save_images.save_images( bicubic , OUTPUT_PATH+'/samples_{}_bicubic.png'.format(it))




    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    #tf.train.start_queue_runners()
    # Train loop
    #gen = inf_train_gen()
    saver = tf.train.Saver(var_list= tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES) )
    if os.path.exists( CHECKPOINT_PATH+"/srwgan.meta" ):
        saver.restore( session , CHECKPOINT_PATH+"/srwgan" )
    else:
        session.run(tf.global_variables_initializer())
        
    it = global_step.eval
    
    best_cost = 1e10
    while it() < EPOCH_SIZE * NUM_EPOCHS :
        train_batch = lib.read.get_batch( data_train , BATCH_SIZE)

        if (it() < 50) or it() % LOG_STEP == LOG_STEP-1  :
            #print("learning_rate = {}".format( lr.eval()  ))
            val_batch = lib.read.get_batch( data_val , BATCH_SIZE )
            train_gen_cost  = session.run( gen_cost  , feed_dict = { minibatch:train_batch } ) 
            val_gen_cost  = session.run( gen_cost , feed_dict = { minibatch:val_batch } ) 
            if best_cost > val_gen_cost :
                best_cost = val_gen_cost
                saver.save( session , CHECKPOINT_PATH+'/bestsrwgan' )
            s = time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(time.time())) +"epoch "+it()/EPOCH_SIZE+ "iter "+str(it()) + ' train  gen cost {}'.format(  train_gen_cost)
            s += "            val gen cost {}".format(  val_gen_cost )  
            print(s)
            saver.save( session , CHECKPOINT_PATH+'/srwgan')
            generate_image(it())
            
        _ = session.run(gen_train_op,feed_dict={minibatch:train_batch})



      #  lib.plot.tick()
