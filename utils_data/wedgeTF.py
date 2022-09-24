import numpy as np
import tensorflow as tf
from scipy.integrate import quadrature

def one_over_E(z, OMm):
    return 1 / np.sqrt(OMm * (1.0 + z) ** 3 + (1 - OMm))

def multiplicative_factor(z, OMm):
    return (1 / one_over_E(z, OMm) / (1 + z) * quadrature(lambda x: one_over_E(x, OMm), 0, z)[0])

def calculate_k_cube(HII_DIM, chunk_length, cell_size):
    k = np.fft.fftfreq(HII_DIM, d=cell_size)
    k_parallel = np.fft.fftfreq(chunk_length, d=cell_size)
    delta_k = k_parallel[1] - k_parallel[0]
    k_cube = np.meshgrid(k, k, k_parallel)
    return tf.constant(k_cube, dtype = tf.float32), tf.constant(delta_k, dtype = tf.float32)

def calculate_blackman(chunk_length, delta_k):
    bm = np.abs(np.fft.fft(np.blackman(chunk_length))) ** 2
    buffer = delta_k * (np.where(bm / np.amax(bm) <= 1e-10)[0][0] - 1)
    BM = np.blackman(chunk_length)[np.newaxis, np.newaxis, :]
    return tf.constant(BM, dtype = tf.complex64), tf.constant(buffer, dtype = tf.float32)

def wedge_removal_tf(OMm, redshifts, HII_DIM, cell_size, Box, chunk_length=501, blackman=True, MF = None, k_cube_data = None, blackman_data = None,):
    # here we allow to pass multiplicative factor, k_cube_data, blackman_data, just so that it is not constantly re-computed
    
    permute = [2, 0, 1]
    inverse_permute = [1, 2, 0]
    
    if chunk_length % 2 == 0:
        chunk_length += 1
    
    if MF is None:
        MF = tf.constant([multiplicative_factor(z, OMm) for z in redshifts], dtype = tf.float32)
    # Box = tf.constant(Box, dtype = tf.float32)
    Box_uv = tf.transpose(tf.cast(Box, tf.complex64), perm = permute)
    Box_uv = tf.signal.fft2d(Box_uv)
    Box_uv = tf.transpose(Box_uv, perm = inverse_permute)

    if k_cube_data is None:
        k_cube, delta_k = calculate_k_cube(HII_DIM, chunk_length, cell_size)
    else:
        k_cube, delta_k = k_cube_data

    if blackman_data is None:
        BM, buffer = calculate_blackman(chunk_length, delta_k) 
    else:
        BM, buffer = blackman_data

    box_shape = Box_uv.shape
    # Box_final = np.empty(box_shape, dtype=np.float32)
    Box_final = []
    empty_box = tf.zeros(k_cube[0].shape, dtype = tf.complex64)
    Box_uv = tf.concat([empty_box, Box_uv, empty_box], axis=2)

    for i in range(chunk_length, box_shape[-1] + chunk_length):
        t_box = Box_uv[..., i - chunk_length // 2 : i + chunk_length // 2 + 1]
        W = k_cube[2] / (
            tf.math.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2)
            * MF[tf.math.minimum(i - chunk_length // 2 - 1, box_shape[-1] - 1)]
            + buffer)

        w = tf.cast(tf.math.logical_or(W < -1.0, W > 1.0), tf.complex64)
        # w = cp.array(W[i + chunk_length - 1])
        if blackman is True:
            t_box = t_box * BM
        Box_final.append(tf.math.real(tf.signal.ifft3d(tf.signal.fft(t_box) * w)[..., chunk_length // 2]))

    return tf.transpose(tf.convert_to_tensor(Box_final), perm = inverse_permute).numpy()
