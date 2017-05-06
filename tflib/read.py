import numpy as np
import scipy.misc
import time

def get_batch(path_array,  batch_size):

    #st = time.time() 
    images = np.zeros((batch_size,  112, 96 , 3 ), dtype='uint8')

    n = len(path_array)
    mask = np.random.choice( n , batch_size )
    #print(mask)
  #  print(mask.shape)
  #  temp = path_array[mask]

    for i , j  in enumerate(path_array[mask]):
        images[ i ] = scipy.misc.imread(j)
    #print("images.shape" , images.shape)
   # ed = time.time()
   # print( "get_batch using ",ed-st )
    return images
