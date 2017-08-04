import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def parse_single_data(file_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)

    features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    }

    example = tf.parse_single_example(value, features=features)

    image = tf.image.decode_image(example['image_raw'], 3)
    #image = tf.decode_raw( example['img_raw'] , uint8 )
    #image.set_shape([None, None, 3])
    image.set_shape([112,96,3])
    label = tf.cast(example['label'], tf.int32)

    return image, label


def preprocessing(image, image_size, flip_flag ): 
#    image = tf.image.resize_images(image, image_size, tf.image.ResizeMethod.BILINEAR)
    if flip_flag:
        image = tf.image.random_flip_left_right(image)
 #   image = tf.cast( image , tf.float32 )
   
    #Ex = tf.reduce_mean(image)
    #Ex2 = tf.reduce_mean(image**2)
    #variance = Ex2 - Ex**2
    #image_ = ( image - tf.reduce_mean( image ) ) / tf.sqrt(variance)
    
#    image_ = image /127.5 -  1.
    #image_ = image
    return image_


def get_batch(file_queue, image_size, batch_size, n_threads, min_after_dequeue, flip_flag ):

    t_list = []
    for i in range(n_threads):
        image, label = parse_single_data(file_queue)
#        image = preprocessing(image, image_size, flip_flag = flip_flag)
        image.set_shape([image_size[0], image_size[1], 3])
        t_list.append((image, label))

    #batch images
    capacity = min_after_dequeue + (n_threads + 5) * batch_size 
    image_batch, label_batch = tf.train.shuffle_batch_join(
        t_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        name='data')

    return image_batch, label_batch
