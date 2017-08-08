import tensorflow as tf
#from model import *
import tflib.read
import scipy.misc
import numpy as np
from scipy.misc import imsave
from srres import generator
import sys
import os

NAME = sys.argv[1]
OUTPUT_PATH = 'test_output/'+NAME
N_GPUS = 1
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
OUTPUT_DIM = 112*96*3
H=int(112/4)
W=int(96/4)

BATCH_SIZE = 64
#BATCH_SIZE = 994

Generator = generator


DATA_PATH = "dfk_994.list"
#DATA_PATH = 'data.test'
data_path = open( DATA_PATH ).read().split('\n')
data_path.pop(len(data_path)-1)
data_path = np.array( data_path )

CHECKPOINT_PATH = 'checkpoint/'+NAME
if os.path.exists( CHECKPOINT_PATH +'/bestsrresnet.meta' ):
    CHECKPOINT_PATH += '/bestsrresnet'
else:
    CHECKPOINT_PATH += '/srresnet'

images_batch = tf.placeholder( tf.uint8 , shape =(None , H*4 , W * 4 , 3 )  )
gen_costs  , disc_costs , fake_datas , real_datas , bicubic_datas = []  , [] , [] , [] , []
split_x  = tf.split( images_batch , len(DEVICES) , axis = 0  )
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

if not os.path.exists("test_output"):
    os.mkdir("test_output")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
for device_index , device in enumerate(DEVICES):
    with tf.device(device):
        x_idx = split_x[device_index]
        x_idx_pre = tf.cast( x_idx , tf.float32) /127.5 - 1
        x_lr = tf.image.resize_bicubic( x_idx_pre , [ H,W ]  )

        fake_data = Generator( inputs = x_lr )
        fake_datas.append(fake_data)


fake_data = tf.concat( fake_datas , axis = 0  ) 
fake_data = tf.clip_by_value( fake_data , -1 , 1  )
fake_data  = tf.cast( (fake_data+1.0)*(255.99/2) , tf.uint8)

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    loader = tf.train.Saver(var_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ))
    loader.restore( sess , CHECKPOINT_PATH)
    sample_times = int(data_path.shape[0]/BATCH_SIZE) + 1 - (data_path.shape[0]%BATCH_SIZE == 0 )

    for i in range(sample_times):
        _get_batch = tflib.read.get_batch( data_path[i*BATCH_SIZE:] , BATCH_SIZE , random=False )
        samples = sess.run(  fake_data , feed_dict = { images_batch:_get_batch }) 
    #    samples = ((samples+1.)*(255.99/2)).astype('uint8')
        with tf.device('/cpu:0'):
            for j in range( _get_batch.shape[0] ):
                dir_name = OUTPUT_PATH + '/' + data_path[i*BATCH_SIZE+j].split('/')[-2]
                if not os.path.exists( dir_name ):
                    os.mkdir( dir_name )
                imsave(OUTPUT_PATH + '/' + data_path[i*BATCH_SIZE+j].split('/')[-2] + '/'+data_path[i*BATCH_SIZE+j].split('/')[-1] , samples[j])
