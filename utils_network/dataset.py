import tensorflow as tf

from data_generator import LightConeGenerator, LightConeGenerator_Reg#, LightConeGenerator_BigBoy 

class GetDataset:
    def __init__(self):
        self.conf = conf
        self.path = path
        self.data_temp = data_temp
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self):
        

def get_dataset(conf, path, data_temp, batch_size, shuffle=True):
    if(conf.AUGMENT == 'seg'):

    elif(conf.AUGMENT == 'rec'):
    
    elif(conf.AUGMENT == 'seg+rec'):
    
    


        train_generator = LightConeGenerator(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)
        valid_generator = LightConeGenerator(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, tobs=1000, shuffle=True)

        # Define generator functional
        def generator_train():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=True)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys = next(multi_enqueuer.get()) 
            yield batch_xs, batch_ys

        def generator_valid():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=True)
        multi_enqueuer.start(workers=10, max_queue_size=10)
        while True:
            batch_xs, batch_ys = next(multi_enqueuer.get()) 
            yield batch_xs, batch_ys

        # Create dataset from data generator
        train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))
        valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))



# Create data generator from tensorflow.keras.utils.Sequence
train_generator = LightConeGenerator_Reg(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
valid_generator = LightConeGenerator_Reg(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)

# Define generator functional
def generator_train():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=True)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
        #yield(batch_xs, {'output1': batch_ys1, 'output2': batch_ys2})
        yield (batch_xs, {'output_img':batch_ys1, 'output_rec':batch_ys2})
        
def generator_valid():
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=True)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
        #yield(batch_xs, {'output1': batch_ys1, 'output2': batch_ys2})
        yield (batch_xs, {'output_img':batch_ys1, 'output_rec':batch_ys2})

# Create dataset from data generator
train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, {'output_img': tf.float32, 'output_rec': tf.float32}))
valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, {'output_img': tf.float32, 'output_rec': tf.float32}))
