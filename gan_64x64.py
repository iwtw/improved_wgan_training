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
import tflib.read.
import tflib.ops.layernorm
import tflib.plot

# Download 64x64 ImageNet at http://image-net.org/small/download.php and
# fill in the path to the extracted files here!
DATA_DIR = '~/asian-webface-align'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 32
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 2 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 200000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 112*96*3 # Number of pixels in each iamge
DATA_TRAIN = "data.train"
DATA_VAL = "data.val"
H = 28
W = 24

lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # Baseline (G: DCGAN, D: DCGAN)
    return ResnetGenerator, DCGANDiscriminator

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

def FCGenerator(n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)

    output = tf.tanh(output)

    return output

def DCGANGenerator(n_samples, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM])

def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*dim, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, dim, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ResnetGenerator(n_samples, inputs, dim=DIM):

   # output = lib.ops.linear.Linear('Generator.Input', 128, 2*H*W*dim, inputs)
    #print(inputs)
    #assert 1==2
    #output = lib.ops.linear.Linear('Generator.Input' , 3 , 2*H*W*dim , inputs )
    output = lib.ops.conv2d.Conv2D('Generator.Input' , 3 , 2*dim ,1, inputs , stride  = 1   )
    print(output.shape)
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


def MultiplicativeDCGANGenerator(n_samples, noise=None, dim=DIM, bn=True):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim*2, noise)
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

# ! Discriminators

def MultiplicativeDCGANDiscriminator(inputs, dim=DIM, bn=True):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim*2, 5, output, stride=2)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim*2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim*2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim*2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])


def ResnetDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in xrange(5):
        output = ResidualBlock('Discriminator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])


def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, inputs)
    for i in xrange(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])

def DCGANDiscriminator(inputs, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, H, W])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])

def myGenerator( inputs , nonlinearity = tf.nn.relu   ):
    output = tf.reshape( inputs , [-1,3,H,W]  )
    output = lib.ops.conv2d.Conv2D('Generator.Conv1' , 3 , 64 , 5,output , stride = 1  )
    shortcut = nonlinearity( output )
    output = nonlinearity( output )
    block = []
    for i in xrange(16+1):
        block.append(Residual_block('Generator.ResidualBlock'+str(i),64,64,3,block[-1] if i else output ))
    output = lib.ops.conv2d.Conv2D('Generator.Conv2',64,64,3,block[-1],stride=1)
    output = Batchnorm('Generator.BN1',[0,2,3],output)
    output = ouput + shortcut
    output = lib.ops.conv2d.Conv2D('Generator.Conv3' , 64, 256 , 3,output , stride=1)

    return output

Generator, Discriminator = GeneratorAndDiscriminator()

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False )
config.gpu_options.allow_growth=True
with tf.Session(config=config) as session:

    #flipped = tf.image.random_flip_left_right( cropped )
    #minibatch =  tf.cast( tf.train.batch([flipped] , BATCH_SIZE  , capacity = BATCH_SIZE * 30  ) , tf.float32 )
    minibatch = tf.placeholder( tf.uint8 , shape =(BATCH_SIZE , H*4 , W * 4 , 3 )  )
    minibatch_lr = tf.image.resize_bicubic( minibatch , [ H,W ]  )
    x_lr = tf.transpose( minibatch_lr , [0,3,1,2] )
    x = tf.transpose( minibatch , [0,3,1,2])

    #split_real_data_conv  = tf.split( x ,  )
    gen_costs, disc_costs = [],[]

    real_data = tf.reshape(2*((tf.cast( x , tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    fake_data = Generator(BATCH_SIZE , inputs = x_lr )

    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)

    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
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

    gen_cost = tf.add_n(gen_costs)  
    disc_cost = tf.add_n(disc_costs)  

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                       var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    
    # For generating samples
    def generate_image(iteration):
        samples = session.run( fake_data )
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, H*4, W*4)), 'samples_{}.png'.format(iteration))


    #save groud truth images
    #print(  _x.shape )
    session.run(tf.initialize_all_variables())
    tf.train.start_queue_runners()
    print("before real_data saving")
    _x_r = x.eval( {minibatch:get_batch(DATA_TRAIN)} )
    print("real_data saved")
    _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE, 3, H*4, W*4)), 'samples_groundtruth.png')


    # Train loop
    #gen = inf_train_gen()
    for iteration in xrange(ITERS):
        train_batch = get_batch(DATA_TRAIN , BATCH_SIZE)

        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op,feed_dict={minibatch:train_batch})

        # Train critic
        disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _disc_cost,_ = session.run([disc_cost , disc_train_op],feed_dict={minibatch:train_batch})


        if (iteration < 50) or iteration % 100 == 99  :
            val_batch = get_batch(DATA_VAL , BATCH_SIZE )
            train_disc_cost = _disc_cost 
            train_gen_cost = gen_cost.eval({minibatch:train_batch})
            val_disc_cost = disc_cost.eval({minibatch:val_batch})
            val_gen_cost = gen_cost.eval({minibatch:val_batch})
            
            s = time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(time.time())) + "iter "+str(iteration) + ' train disc cost {} gen cost {}'.format( train_disc_cost, train_gen_cost.eval())
            print(s)
            s = "            val disc cost {} gen cost {}".format( val_disc_cost , val_gen_cost )  
            print(s)
            
            generate_image(iteration)

      #  lib.plot.tick()
