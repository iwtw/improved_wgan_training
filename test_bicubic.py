import tensorflow
from model import *
import tflib.read
import scipy.misc
import numpy as np
from scipy.misc import imsave
import sys


OUTPUT_PATH = 'test_output/'+"bicubic"
N_GPUS = 2
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
OUTPUT_DIM = 112*96*3

BATCH_SIZE = 64
SAMPLE_TIMES = 100
#BATCH_SIZE = 994

Generator = ResnetGenerator


minibatch = tf.placeholder( tf.uint8 , shape =(BATCH_SIZE , H*4 , W * 4 , 3 )  )
#DATA_PATH = "dfk_data.test"
DATA_PATH = 'data.test'
data_path = open( DATA_PATH ).read().split('\n')
data_path.pop(len(data_path)-1)
data_path = np.array( data_path )

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
        minibatch_bicubic = tf.image.resize_bicubic( minibatch_lr , [H*4,W*4] )
        bicubic_datas.append(minibatch_bicubic)

bicubic_data = tf.concat( bicubic_datas , axis = 0  )
bicubic_data = tf.clip_by_value( bicubic_data ,  0 , 255)

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    for i in xrange(SAMPLE_TIMES):
        _get_batch = lib.read.get_batch( data_path[i*BATCH_SIZE:] , BATCH_SIZE , random=False )
        samples = sess.run(  bicubic_data , feed_dict = { minibatch:_get_batch }) 
        samples = samples.astype('uint8')
        with tf.device('/cpu:0'):
            for j in xrange(BATCH_SIZE):
                dir_name = OUTPUT_PATH + '/' + data_path[i*BATCH_SIZE+j].split('/')[-2]
                if not os.path.exists( dir_name ):
                    os.mkdir( dir_name )
                imsave(OUTPUT_PATH + '/' + data_path[i*BATCH_SIZE+j].split('/')[-2] + '/'+data_path[i*BATCH_SIZE+j].split('/')[-1] , samples[j])
