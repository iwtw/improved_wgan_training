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
from model import *

TEST_SPEED = False
TENSORFLOW_READ = False

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("output"):
    os.mkdir("output")

P = float(sys.argv[2])
CRITIC_ITERS = 5 # How many its to train the critic for
N_GPUS = 2 # Number of GPUs
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
BATCH_SIZE = N_GPUS * 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 200000 # How many its to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 112*96*3 # Number of pixels in each iamge
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

lib.print_model_settings(locals().copy())



Generator, Discriminator = GeneratorAndDiscriminator()

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False )
config.gpu_options.allow_growth=True
with tf.Session(config=config) as session:
    

    if TENSORFLOW_READ:
        with tf.device('/cpu:0'):
            file_names=open(DATA_TRAIN,'r').read().split('\n')
            file_names.pop( len(file_names) -1 )
            steps_per_epoch = len(file_names) / BATCH_SIZE
            #random.shuffle(file_names)
            filename_queue=tf.train.string_input_producer(file_names)
            reader=tf.WholeFileReader()
            _,value=reader.read(filename_queue)
            image=tf.image.decode_jpeg(value)
            cropped=tf.random_crop(image,[ H *4, W*4,3])
            flipped = tf.image.random_flip_left_right( cropped )
            minibatch =   tf.train.batch([flipped] , BATCH_SIZE  , capacity = BATCH_SIZE * 30  ) 
    else:
        minibatch = tf.placeholder( tf.uint8 , shape =(BATCH_SIZE , H*4 , W * 4 , 3 )  )


    gen_costs  , disc_costs , fake_datas , real_datas , bicubic_datas = []  , [] , [] , [] , []
    split_minibatch  = tf.split( minibatch , len(DEVICES) , axis = 0  )
    for device_index , device in enumerate(DEVICES):
        with tf.device(device):
            minibatch_idx = split_minibatch[device_index]
            minibatch_lr = tf.image.resize_bicubic( minibatch_idx , [ H,W ]  )
            minibatch_bicubic = tf.image.resize_bicubic( minibatch_lr, [H*4 , W*4] )
            minibatch_bicubic = tf.clip_by_value( minibatch_bicubic , 0 ,255 )
            x_lr = tf.transpose( minibatch_lr , [0,3,1,2] )
            x = tf.transpose( minibatch_idx , [0,3,1,2])
            x_bicubic = tf.transpose( minibatch_bicubic , [0,3,1,2] )


            real_data = tf.reshape(2*((tf.cast( x , tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE/len(DEVICES) , inputs = x_lr )


            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            gen_content_cost = tf.losses.mean_squared_error( real_data , fake_data )
            gen_adv_cost = -tf.reduce_mean(disc_fake)
            gen_cost = gen_content_cost + P * gen_adv_cost
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES),1], 
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_cost += LAMBDA*gradient_penalty

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)
            real_datas.append(real_data) 
            fake_datas.append(fake_data)
            bicubic_datas.append(x_bicubic)

    gen_cost = tf.add_n(gen_costs)/len(DEVICES)  
    disc_cost = tf.add_n(disc_costs)/len(DEVICES)  
    real_data = tf.concat( real_datas , axis = 0  )
    fake_data = tf.concat( fake_datas , axis = 0  ) 
    bicubic_data = tf.concat( bicubic_datas , axis = 0  )



    global_step = tf.Variable( initial_value = 0 , dtype = tf.int32 , trainable=0 ,name = 'global_step')
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True , global_step = global_step)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,                                                                  var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    
    # For generating samples
    def generate_image(it):
        _get_batch = lib.read.get_batch( data_train , BATCH_SIZE )
        real_samples , samples  , bicubic_samples = session.run( [real_data , fake_data ,  bicubic_data ] , feed_dict = { minibatch:_get_batch }) 

        samples = ((samples+1.)*(255.99/2)).astype('uint8')
        samples = samples.reshape((BATCH_SIZE, 3, H*4, W*4))
        real_samples = ((real_samples+1.)*(255.99/2)).astype('uint8')
        real_samples = real_samples.reshape((BATCH_SIZE, 3, H*4, W*4))
        bicubic_samples = bicubic_samples.astype('uint8')
        bicubic_samples = bicubic_samples.reshape((BATCH_SIZE , 3 , H*4 , W*4))
        
        with tf.device('/cpu:0'):
            lib.save_images.save_images( samples , OUTPUT_PATH+'/samples_{}_gen.png'.format(it))
            lib.save_images.save_images( real_samples , OUTPUT_PATH+'/samples_{}_real.png'.format(it))
            lib.save_images.save_images( bicubic_samples , OUTPUT_PATH+'/samples_{}_bicubic.png'.format(it)  )



    #session.run(tf.global_variables_initializer())
    #session.run(tf.local_variables_initializer())
    #tf.train.start_queue_runners()
    # Train loop
    #gen = inf_train_gen()
    saver = tf.train.Saver(var_list= tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES) )
    if os.path.exists( CHECKPOINT_PATH+"/srwgan.meta" ):
        saver.restore( session , CHECKPOINT_PATH+"/srwgan" )
    else:
        session.run(tf.global_variables_initializer())
        
    it = global_step.eval
    
    while it() < ITERS :
        train_batch = lib.read.get_batch( data_train , BATCH_SIZE)

        if (it() < 50) or it() % 100 == 99  :
            val_batch = lib.read.get_batch( data_val , BATCH_SIZE )
            train_gen_cost ,train_disc_cost = session.run( [gen_cost , disc_cost] , feed_dict = { minibatch:train_batch } ) 
            val_gen_cost , val_disc_cost = session.run( [gen_cost , disc_cost ] , feed_dict = { minibatch:val_batch } ) 
            s = time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(time.time())) + "iter "+str(it()) + ' train disc cost {} gen cost {}'.format( train_disc_cost, train_gen_cost)
            s += "            val disc cost {} gen cost {}".format( val_disc_cost , val_gen_cost )  
            print(s)
            saver.save( session , CHECKPOINT_PATH+'/srwgan')
            generate_image(it())
            
        disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _ = session.run( disc_train_op,feed_dict={minibatch:train_batch})
        _ = session.run(gen_train_op,feed_dict={minibatch:train_batch})



      #  lib.plot.tick()
