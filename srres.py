import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.save_images
#import tflib.read 
import data_input
import os
import sys
import vgg19
import utils
#import vgg

TEST_SPEED = False
TENSORFLOW_READ = False
if not os.path.exists("checkpoint"): os.mkdir("checkpoint")
if not os.path.exists("output"):
    os.mkdir("output")

#LEARNING_RATE=tf.Variable( 5e-5 )
LOG_STEP=1000
P =float( sys.argv[2] )
DIM = 32 
N_GPUS = 2
BATCH_SIZE = N_GPUS * 128 
N_EPOCHS = 25
EPOCH_SIZE = int( 917719 / BATCH_SIZE )
LAMBDA = 10 
OUTPUT_DIM = 112*96*3 
DATA_TRAIN = ["/mnt/data-set-1/dataset-tfrecord/asian-webface-lcnn-40/train_0{}.tfrecord".format(str(i).zfill(2)) for i in range(15)]
#DATA_VAL = "data.val"

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

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def block(inputs , kernel_size ):
    
    shortcut = inputs
    dim = inputs.get_shape().dims[-1].value
    conv1 = utils.conv2d( inputs , outputs_dim = dim , kernel_size = kernel_size , stride = 1  )
    bn1 = tf.contrib.layers.batch_norm( conv1 , activation_fn = tf.nn.relu )
    conv2 = utils.conv2d( inputs , outputs_dim = dim , kernel_size = kernel_size , stride = 1  )
    bn2 = tf.contrib.layers.batch_norm( conv2 , activation_fn = None  )
    outputs = shortcut + bn2
    return outputs

def subpixel_conv2d( inputs , outputs_dim , kernel_size , stride  , block_size ):
    conv = utils.conv2d( inputs , outputs_dim , kernel_size , stride  )
    subpixel = tf.depth_to_space( conv , block_size )
    return subpixel

def generator( inputs  ):
    conv1 = utils.conv2d( inputs , outputs_dim = DIM , kernel_size = 9 , stride = 1  , he_init = True , activation_fn = tf.nn.relu )
    
    conv2_x = conv1
    for i in range(16):
        conv2_x = block( conv2_x , kernel_size = 3  )
    conv3 = utils.conv2d( conv2_x , outputs_dim = DIM , kernel_size = 3 , stride = 1 )
    bn3 = tf.contrib.layers.batch_norm( conv3  )
    bn3 += conv1

    upsample1 = subpixel_conv2d( bn3 , outputs_dim = DIM , kernel_size = 3 , stride = 1 , block_size = 2   )
    upsample1 = tf.nn.relu( upsample1 )
    upsample2 = subpixel_conv2d( upsample1 , outputs_dim = DIM , kernel_size = 3 , stride = 1 , block_size = 2 )
    upsample2 = tf.nn.relu( upsample2 )
    outputs = utils.conv2d( upsample2 , outputs_dim = 3 , kernel_size = 9 , stride = 1  )
    return outputs
    
    
Generator = generator




    

def build_graph( data ):
    #image_batch = tf.placeholder( tf.uint8 , shape =(BATCH_SIZE , H*4 , W * 4 , 3 )  )
    file_queue = tf.train.string_input_producer( data )
    image_batch , label_batch  = data_input.get_batch( file_queue , (112,96) , BATCH_SIZE , n_threads = 4 , min_after_dequeue = 0 , flip_flag = True )
    # image_batch ranges [0,255] , dtype = tf.uint8
    

    #preprocessing
    with tf.device("/gpu:0"):
       # image_batch_pre = tf.image.random_flip_left_right(image_batch)
        image_batch_pre = tf.cast( image_batch, tf.float32 )
        image_batch_pre =  image_batch_pre /127.5 -1.

        gen_losses   , x_gens  , x_bicubics = [] ,  [] , [] 
        content_losses , vgg_losses = [] , []
        split_image_batch_pre  = tf.split( image_batch_pre , len(DEVICES) , axis = 0  )

    for device_index , device in enumerate(DEVICES):
        with tf.device(device):
            print("device : ", device)
            x = split_image_batch_pre[device_index]
            x_lr = tf.image.resize_bicubic( x , [ H,W ]  )
            x_bicubic =  tf.image.resize_bicubic ( x_lr , [H*4,W*4] ) 
            x_gen = Generator( inputs = x_lr  )

            vgg_inputs = tf.concat(  [ x , x_gen ] , axis = 0 ) 
            vgg_inputs = (vgg_inputs + 1.0)  *127.5
            vgg=vgg19.Vgg19()
            vgg.build( vgg_inputs )
            fmap=tf.split(vgg.conv2_2,2)
            vgg_loss = tf.losses.mean_squared_error( fmap[0] , fmap[1] )
            gen_content_loss = tf.losses.mean_squared_error( x , x_gen )
            gen_loss = (1 -P ) * gen_content_loss + P * vgg_loss 

            gen_losses.append(gen_loss)
            vgg_losses.append(vgg_loss)
            x_bicubics.append(x_bicubic)
            x_gens.append(x_gen)
            content_losses.append(gen_content_loss)

    gen_loss = tf.add_n(gen_losses) * ( 1.0 / len(DEVICES) ) * (1.0 / BATCH_SIZE)
    vgg_loss = tf.add_n(vgg_losses ) * ( 1.0 / len(DEVICES)) * (1.0 / BATCH_SIZE)
    content_loss = tf.add_n(content_losses)  * (1.0 / len(DEVICES)) * (1.0 / BATCH_SIZE)
    x_gen = tf.concat( x_gens , axis = 0 ) 
    x_bicubic = tf.concat( x_bicubics , axis = 0 )
    tf.summary.scalar("gen_loss" , gen_loss)
    tf.summary.scalar("vgg_loss" , vgg_loss)
    tf.summary.scalar("content_loss" , content_loss )

    def convert(x):
        # x ranges[-1,1] , dtype = tf.float32 
        # returns: ranges [0,255] , dtype = tf.uint8 
        x = tf.clip_by_value( x , -1,1 )
        x = (x+1.0) *(255.99/2)
        x = tf.cast(x , tf.uint8)
        return x
        
    x_gen_outputs = convert(x_gen) 
    x_bicubic_outputs = convert(x_bicubic)
    outputs = tf.concat( [image_batch , x_gen_outputs , x_bicubic_outputs] , axis = 0  )

    return outputs , gen_loss

    
def generate_image(sess,it,outputs):
    # For generating samples
    lib.save_images.save_images( outputs ,  5 ,   OUTPUT_PATH+'/samples_{}.png'.format(it))

def train(outputs,gen_loss):
    global_step = tf.Variable( initial_value = 0 , dtype = tf.int32 , trainable = 0 ,name = 'global_step')
    boundaries = [ 5 * EPOCH_SIZE ,  10 * EPOCH_SIZE , 15 * EPOCH_SIZE , 20 * EPOCH_SIZE]
    lrs = [ 1e-3 , 1e-4 , 5e-5 , 1e-5 , 1e-6 ]
    lr = tf.train.piecewise_constant( global_step , boundaries , lrs  )
    gen_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(gen_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), colocate_gradients_with_ops=True , global_step = global_step)
    config = tf.ConfigProto(allow_soft_placement=True )
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(var_list= tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES) )
        if os.path.exists( CHECKPOINT_PATH+"/srresnet.meta" ):
            saver.restore( sess , CHECKPOINT_PATH+"/srresnet" )
        else:
            sess.run(tf.global_variables_initializer())
            
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator(  )
        threads = tf.train.start_queue_runners( sess )
        it = global_step.eval
        
        best_loss = 1e10
        
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("summarylog/{}/".format(NAME),sess.graph)
        while it() < EPOCH_SIZE * N_EPOCHS :
     #       train_batch = lib.read.get_batch( data_train , BATCH_SIZE)
     #       vgg_inputs_ = sess.run(vgg_inputs , feed_dict = {image_batch:train_batch})
    #        print(np.min(vgg_inputs_) , np.max(vgg_inputs_))

            if (it() < 50) or it() % LOG_STEP == LOG_STEP-1  :
                #print("learning_rate = {}".format( lr.eval()  ))
                #val_batch = lib.read.get_batch( data_val , BATCH_SIZE )
                train_gen_loss , log   = sess.run( [ gen_loss , merged_summary ] ) 
                #val_gen_loss  = sess.run( gen_loss , feed_dict = { image_batch:val_batch } ) 
                if best_loss > train_gen_loss :
                    best_loss = train_gen_loss
                    saver.save( sess , CHECKPOINT_PATH+'/bestsrresnet' )
                s = time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(time.time())) +"epoch "+ str( int( it()/EPOCH_SIZE) )+ " iter "+str(it()) + ' train  gen loss {}'.format(  train_gen_loss)
                #s += "            val gen loss {}".format(  val_gen_loss )  
                print(s)
                saver.save( sess , CHECKPOINT_PATH+'/srresnet')
                generate_image(sess,it(),sess.run(outputs))
                writer.add_summary( log , it() )
                
            _ = sess.run(gen_train_op)
        coord.request_stop()
        coord.join(threads)

def main(_):
    outputs , gen_loss = build_graph(DATA_TRAIN)
    train(outputs,gen_loss)

if __name__ == "__main__":
    tf.app.run(main)
