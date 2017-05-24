import tensorflow
from model import *
import tflib.read
import scipy.misc
import numpy as np
from scipy.misc import imsave
import sys

NAME = sys.argv[1]
OUTPUT_PATH = 'test_output/'+NAME
N_GPUS = 2
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
OUTPUT_DIM = 112*96*3

BATCH_SIZE = 64
SAMPLE_TIMES = 100
#BATCH_SIZE = 994

Generator = ResnetGenerator


minibatch = tf.placeholder( tf.uint8 , shape =(BATCH_SIZE , H*4 , W * 4 , 3 )  )
#DATA_PATH = "dfk_data.test"
DATA_PATH = 'data.val'
data_path = open( DATA_PATH ).read().split('\n')
data_path.pop(len(data_path)-1)
data_path = np.array( data_path )
CHECKPOINT_PATH = 'checkpoint/'+NAME
if os.path.exists( CHECKPOINT_PATH +'/bestsrwgan.meta' ):
    CHECKPOINT_PATH += '/bestsrwgan'
else:
    CHECKPOINT_PATH += '/srwgan'

gen_costs  , disc_costs , fake_datas , real_datas , bicubic_datas = []  , [] , [] , [] , []
split_minibatch  = tf.split( minibatch , len(DEVICES) , axis = 0  )
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

if not os.path.exists("test_output"):
    os.mkdir("test_output")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
for device_index , device in enumerate(DEVICES):
    with tf.device(device):
        minibatch_idx = split_minibatch[device_index]
        minibatch_lr = tf.image.resize_bicubic( minibatch_idx , [ H,W ]  )
        x_lr = tf.transpose( minibatch_lr , [0,3,1,2] )

        fake_data = Generator(BATCH_SIZE/len(DEVICES) , inputs = x_lr )
        fake_datas.append(fake_data)


fake_data = tf.concat( fake_datas , axis = 0  ) 
fake_data = tf.reshape( fake_data , [BATCH_SIZE , 3 , 112,96 ] )
fake_data = tf.transpose( fake_data,   [0,2,3,1] )

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    loader = tf.train.Saver(var_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ))
    loader.restore( sess , CHECKPOINT_PATH)
    for i in xrange(SAMPLE_TIMES):
        _get_batch = lib.read.get_batch( data_path[i*BATCH_SIZE:] , BATCH_SIZE , random=False )
        samples = sess.run(  fake_data , feed_dict = { minibatch:_get_batch }) 
        samples = ((samples+1.)*(255.99/2)).astype('uint8')
        with tf.device('/cpu:0'):
            for j in xrange(BATCH_SIZE):
                imsave(OUTPUT_PATH + '/' + data_path[i*BATCH_SIZE+j].split('/')[-1] , samples[j])
