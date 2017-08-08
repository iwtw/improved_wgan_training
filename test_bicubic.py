import tensorflow as tf
#from model import *
import tflib.read
import scipy.misc
import numpy as np
from scipy.misc import imsave
import sys
import os

NAME = sys.argv[1]
OUTPUT_PATH = 'test_output/'+NAME
N_GPUS = 2
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
OUTPUT_DIM = 112*96*3
H=int(112/4)
W=int(96/4)

BATCH_SIZE = 64



DATA_PATH = "dfk_994.txt"
#DATA_PATH = 'data.test'
data_path = open( DATA_PATH ).read().split('\n')
data_path.pop(len(data_path)-1)
data_path = np.array( data_path )

images_batch = tf.placeholder( tf.uint8 , shape =(None , H*4 , W * 4 , 3 )  )
gen_costs  , disc_costs , fake_datas , real_datas , bicubic_datas = []  , [] , [] , [] , []
split_x  = tf.split( images_batch , len(DEVICES) , axis = 0  )
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

if not os.path.exists("test_output"):
    os.mkdir("test_output")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

x_bicubics = []
for device_index , device in enumerate(DEVICES):
    with tf.device(device):
        x_idx = split_x[device_index]
        x_idx_pre = tf.cast( x_idx , tf.float32) /127.5 - 1
        x_lr = tf.image.resize_bicubic( x_idx_pre , [ H,W ]  )
        x_bicubic = tf.image.resize_bicubic( x_lr , [H*4,W*4] )

        x_bicubics.append(x_bicubic)

x_bicubic = tf.concat( x_bicubics , axis = 0  )
#x_bicubic = tf.clip_by_value( x_bicubic , -1 , 1 )
def convert(x):
    x = tf.cast ( (x + 1.) *(255.99/2) ,  tf.uint8 ) 
    return x
x_bicubic = convert( x_bicubic )

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    sample_times = int(data_path.shape[0]/BATCH_SIZE) + 1 - (data_path.shape[0]%BATCH_SIZE == 0 )

    for i in range(sample_times):
        _get_batch = tflib.read.get_batch( data_path[i*BATCH_SIZE:] , BATCH_SIZE , random=False )
        samples = sess.run(  x_bicubic , feed_dict = { images_batch:_get_batch }) 
    #    samples = ((samples+1.)*(255.99/2)).astype('uint8')
        with tf.device('/cpu:0'):
            for j in range( _get_batch.shape[0] ):
                dir_name = OUTPUT_PATH + '/' + data_path[i*BATCH_SIZE+j].split('/')[-2]
                if not os.path.exists( dir_name ):
                    os.mkdir( dir_name )
                imsave(OUTPUT_PATH + '/' + data_path[i*BATCH_SIZE+j].split('/')[-2] + '/'+data_path[i*BATCH_SIZE+j].split('/')[-1] , samples[j])
