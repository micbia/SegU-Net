import tensorflow as tf, os

print('Number nodes:', os.environ['SLURM_JOB_NUM_NODES'])
print('Number GPUs Available:', len(tf.config.list_physical_devices('GPU')))