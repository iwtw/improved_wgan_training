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

TEST_SPEED = False
TENSORFLOW_READ = False

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("output"):
    os.mkdir("output")
# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!
DATA_DIR = '~/asian-webface-align'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')


MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 64 
CRITIC_ITERS = 5 # How many its to train the critic for
N_GPUS = 2 # Number of GPUs
BATCH_SIZE = 32 # Batch size. Must be a multiple of N_GPUS
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

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # Baseline (G: DCGAN, D: DCGAN)
    return ResnetGenerator, ResnetDiscriminator

    # No BN and constant number of filts in G
    # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

    # 512-dim 4-layer ReLU MLP G
    # return FCGenerator, DCGANDiscriminator

    # No normalization anywhere
    # return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

    # Gated multiplicative nonlinearities everywhere
    # return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

    # tanh nonlinearities everywhere
    # return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), \
    #        functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

    # 101-layer ResNet G and D
    # return ResnetGenerator, ResnetDiscriminator

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
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
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

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
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
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

# ! Generators
def ResnetGenerator(n_samples, inputs, dim=DIM):

   # output = lib.ops.linear.Linear('Generator.Input', 128, 2*H*W*dim, inputs)
    #output = lib.ops.linear.Linear('Generator.Input' , 3 , 2*H*W*dim , inputs )
    output = lib.ops.conv2d.Conv2D('Generator.Input' , 3 , 2*dim ,1, inputs , stride  = 1   )
    output = tf.reshape( output , [-1, 2*dim, H, W ])

    for i in xrange(6):
        output = ResidualBlock('Generator.{}x{}_{}'.format(H,W,i), 2*dim, 2*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up1', 2*dim, 1*dim, 3, output, resample='up')
    for i in xrange(6):
        output = ResidualBlock('Generator.{}x{}_{}'.format(2*H,2*W,i), 1*dim, 1*dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up2', 1*dim, dim/2, 3, output, resample='up')
    for i in xrange(5):
        output = ResidualBlock('Generator.{}x{}_{}'.format(4*H,4*W,i), dim/2, dim/2, 3, output, resample=None)

    output = lib.ops.conv2d.Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, OUTPUT_DIM])



# ! Discriminators



def ResnetDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 112, 96])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in xrange(5):
        output = ResidualBlock('Discriminator.112x96_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.56x48_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.28x24_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.14x12_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.7x6_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 7*6*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 7*6* 8*dim, 512, output)

    return tf.reshape(output/5. , [-1])




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


    gen_costs  , disc_costs , fake_datas , real_datas = []  , [] , [] , []
    split_minibatch  = tf.split( minibatch , len(DEVICES) , axis = 0  )
    for device_index , device in enumerate(DEVICES):
        with tf.device(device):
            minibatch_idx = split_minibatch[device_index]
            minibatch_lr = tf.image.resize_bicubic( minibatch_idx , [ H,W ]  )
            x_lr = tf.transpose( minibatch_lr , [0,3,1,2] )
            x = tf.transpose( minibatch_idx , [0,3,1,2])


            real_data = tf.reshape(2*((tf.cast( x , tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE/len(DEVICES) , inputs = x_lr )


            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            gen_cost = -tf.reduce_mean(disc_fake)
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

    gen_cost = tf.add_n(gen_costs)/len(DEVICES)  
    disc_cost = tf.add_n(disc_costs)/len(DEVICES)  
    real_data = tf.concat( real_datas , axis = 0  )
    fake_data = tf.concat( fake_datas , axis = 0  ) 



    global_step = tf.Variable( initial_value = 0 , dtype = tf.int32 , trainable=0 ,name = 'global_step')

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True , global_step = global_step)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                       var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    
    # For generating samples
    def generate_image(it):

        _get_batch = lib.read.get_batch( data_train , BATCH_SIZE )
        real_samples , samples = session.run( [real_data , fake_data ] , feed_dict = { minibatch:_get_batch }) 

        samples = ((samples+1.)*(255.99/2)).astype('uint8')
        samples = samples.reshape((BATCH_SIZE, 3, H*4, W*4))
        real_samples = ((real_samples+1.)*(255.99/2)).astype('uint8')
        real_samples = real_samples.reshape((BATCH_SIZE, 3, H*4, W*4))
        
        with tf.device('/cpu:0'):
            lib.save_images.save_images( samples , OUTPUT_PATH+'/samples_{}_gen.png'.format(it))
            lib.save_images.save_images( real_samples , OUTPUT_PATH+'/samples_{}_real.png'.format(it))



    if TEST_SPEED:
        x_time = np.zeros(10)
        whole_time = np.zeros(10)
        
    
        for j in xrange(10):
            if TENSORFLOW_READ:
                session.run(tf.global_variables_initializer())
                session.run(tf.local_variables_initializer())
                tf.train.start_queue_runners()
                st = time.time()
                session.run( x  )
                ed = time.time()
                x_time[j] =  ed - st
                st = time.time()
                for i in xrange(CRITIC_ITERS):
                    session.run ( disc_train_op )
                session.run( gen_train_op  )
                ed = time.time()
                whole_time[j]= ed - st
            else:
                session.run(tf.global_variables_initializer())
                st = time.time()
                a = lib.read.get_batch( data_train ,BATCH_SIZE)
                session.run( x , feed_dict={minibatch:a}  )
                ed = time.time()
                x_time[j] = ed - st 

                st = time.time()
                a = lib.read.get_batch( data_train ,BATCH_SIZE)
                for i in xrange(CRITIC_ITERS):
                    session.run ( disc_train_op ,  feed_dict={minibatch:a})
                session.run( gen_train_op , feed_dict = {minibatch:a} )
                ed = time.time()
                whole_time[j] = ed - st 
        print("x:time {} , whole time {}".format(np.mean(x_time), np.mean(whole_time)))

        assert 1==2


    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
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
