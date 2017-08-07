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
#import tflib.read 
import data_input
import tflib.ops.layernorm
import tflib.plot
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
N_GPUS = 1
BATCH_SIZE = N_GPUS * 108 
N_EPOCHS = 25
EPOCH_SIZE = int( 690223 / BATCH_SIZE )
LAMBDA = 10 
OUTPUT_DIM = 112*96*3 
DATA_TRAIN = ["/mnt/tfrecord/asian-webface_train.tfrecord"]
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
 #   output = tf.tanh(output )

    return tf.reshape(output, [-1, OUTPUT_DIM])


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

def generator( inputs , dim ):
    conv1 = utils.conv2d( inputs , outputs_dim = dim , kernel_size = 9 , stride = 1  , he_init = True , activation_fn = tf.nn.relu )
    
    conv2_x = conv1
    for i in range(16):
        conv2_x = block( conv2_x , kernel_size = 3  )
    conv3 = utils.conv2d( conv2_x , outputs_dim =dim , kernel_size = 3 , stride = 1 )
    bn3 = tf.contrib.layers.batch_norm( conv3  )
    bn3 += conv1

    upsample1 = subpixel_conv2d( bn3 , outputs_dim = dim , kernel_size = 3 , stride = 1 , block_size = 2   )
    upsample1 = tf.nn.relu( upsample1 )
    upsample2 = subpixel_conv2d( upsample1 , outputs_dim = dim , kernel_size = 3 , stride = 1 , block_size = 2 )
    upsample2 = tf.nn.relu( upsample2 )
    outputs = utils.conv2d( upsample2 , outputs_dim = 3 , kernel_size = 9 , stride = 1  )
    return outputs
    
    
Generator = generator



DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

config = tf.ConfigProto(allow_soft_placement=True )
config.gpu_options.allow_growth=True
with tf.Session(config=config) as session:
    

    #image_batch = tf.placeholder( tf.uint8 , shape =(BATCH_SIZE , H*4 , W * 4 , 3 )  )
    file_queue = tf.train.string_input_producer( DATA_TRAIN )
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
            x = split_image_batch_pre[device_index]
            x_lr = tf.image.resize_bicubic( x , [ H,W ]  )
            x_bicubic =  tf.image.resize_bicubic ( x_lr , [H*4,W*4] ) 
            
           
            
    #        x_lr = tf.transpose( image_batch_lr , [0,3,1,2] )
    #        x = tf.transpose( image_batch_idx , [0,3,1,2])
    #        x_bicubic = tf.transpose( image_batch_bicubic , [0,3,1,2] )

            x_gen = Generator( inputs = x_lr , dim = DIM )

            

            vgg_inputs = tf.concat(  [ x , x_gen ] , axis = 0 ) 
            vgg_inputs = (vgg_inputs + 1.0)  *127.5
            vgg=vgg19.Vgg19()
            vgg.build( vgg_inputs )
            fmap=tf.split(vgg.conv2_2,2)
            vgg_loss = tf.losses.mean_squared_error( fmap[0] , fmap[1] )
            gen_content_loss = tf.losses.mean_squared_error( x , x_gen )
            gen_loss = (1 -P ) * gen_content_loss + P * vgg_loss 

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES),1], 
                minval=0.,
                maxval=1.
            )

            gen_losses.append(gen_loss)
            vgg_losses.append(vgg_loss)
            x_bicubics.append(x_bicubic)
            x_gens.append(x_gen)
            content_losses.append(gen_content_loss)

    with tf.device("/gpu:0"):
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
        x = tf.clip_by_value( x_bicubic , -1,1 )
        x = (x+1.0) *(255.99/2)
        x = tf.cast(x , tf.uint8)
        return x
        
    with tf.device("/gpu:0"):
        x_gen_outputs = convert(x_gen) 
        x_bicubic_outputs = convert(x_bicubic)
        outputs = tf.concat( [image_batch , x_gen_outputs , x_bicubic_outputs] , axis = 0  )



    global_step = tf.Variable( initial_value = 0 , dtype = tf.int32 , trainable = 0 ,name = 'global_step')
    #boundaries = [ epoch_size * 6 , epoch_size * 11 , epoch_size * 16 ]
    boundaries = [ 5 * EPOCH_SIZE ,  10 * EPOCH_SIZE , 15 * EPOCH_SIZE , 20 * EPOCH_SIZE]
    lrs = [ 1e-3 , 1e-4 , 5e-5 , 1e-5 , 1e-6 ]
    lr = tf.train.piecewise_constant( global_step , boundaries , lrs  )
    gen_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(gen_loss,
                                      var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), colocate_gradients_with_ops=True , global_step = global_step)

    
    # For generating samples
    def generate_image(sess,it):

        outputs_ = sess.run( outputs )
        lib.save_images.save_images( outputs_ ,  5 ,   OUTPUT_PATH+'/samples_{}.png'.format(it))




    
    saver = tf.train.Saver(var_list= tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES) )
    if os.path.exists( CHECKPOINT_PATH+"/srresnet.meta" ):
        saver.restore( session , CHECKPOINT_PATH+"/srresnet" )
    else:
        session.run(tf.global_variables_initializer())
        
    session.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator(  )
    threads = tf.train.start_queue_runners( session )
    it = global_step.eval
    
    best_loss = 1e10
    
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("summarylog/{}/".format(NAME),session.graph)
    while it() < EPOCH_SIZE * N_EPOCHS :
 #       train_batch = lib.read.get_batch( data_train , BATCH_SIZE)
 #       vgg_inputs_ = session.run(vgg_inputs , feed_dict = {image_batch:train_batch})
#        print(np.min(vgg_inputs_) , np.max(vgg_inputs_))

        if (it() < 50) or it() % LOG_STEP == LOG_STEP-1  :
            #print("learning_rate = {}".format( lr.eval()  ))
            #val_batch = lib.read.get_batch( data_val , BATCH_SIZE )
            train_gen_loss , log   = session.run( [ gen_loss , merged_summary ] ) 
            #val_gen_loss  = session.run( gen_loss , feed_dict = { image_batch:val_batch } ) 
            if best_loss > train_gen_loss :
                best_loss = train_gen_loss
                saver.save( session , CHECKPOINT_PATH+'/bestsrresnet' )
            s = time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(time.time())) +"epoch "+ str( it()/EPOCH_SIZE) + "iter "+str(it()) + ' train  gen loss {}'.format(  train_gen_loss)
            #s += "            val gen loss {}".format(  val_gen_loss )  
            print(s)
            saver.save( session , CHECKPOINT_PATH+'/srresnet')
            generate_image(session,it())
            writer.add_summary( log , it() )
            
        _ = session.run(gen_train_op)
    coord.request_stop()
    coord.join(threads)



      #  lib.plot.tick()
