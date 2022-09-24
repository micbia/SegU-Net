import tensorflow.keras.backend as K
import tensorflow as tf, numpy as np
from utils_network.metrics import balanced_cross_entropy, sigmoid_balanced_cross_entropy_with_logits

#y_true = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
y_true = np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.1]]])
y_pred = np.array([[[0.0, 0.8, 0.1], [0.3, 0.1, 0.1], [0.1, 0.2, 0.15]], [[0.1, 0.8, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])

def balanced_binary_crossentropy(target, output, from_logits=False):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)

    beta = tf.maximum(tf.reduce_mean(target, axis=(1,2,3)), K.epsilon())[...,tf.newaxis, tf.newaxis, tf.newaxis]
    
    if from_logits:
        y_pred = tf.clip_by_value(output, K.epsilon(), 1 - K.epsilon())
        return sigmoid_balanced_cross_entropy_with_logits(logits=target, labels=K.log(y_pred/(1-y_pred)), beta=beta)

    epsilon_ = tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = beta * target * tf.math.log(output + K.epsilon())
    bce += (1-beta) * (1 - target) * tf.math.log(1 - output + K.epsilon())
    #return -bce
    return tf.reduce_mean(-bce)

def balanced_cross_entropy(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    print(y_true.ndim)
    #beta = tf.maximum(tf.reduce_mean(1-y_true), tf.keras.backend.epsilon())
    beta = tf.maximum(tf.reduce_mean(y_true, axis=(1,2)), tf.keras.backend.epsilon())
    #beta = tf.maximum(K.mean(y_true), tf.keras.backend.epsilon())
    print(beta)
    #y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    #y_pred = K.log(y_pred / (1 - y_pred))
    y_pred = 1./ (1.+K.exp(-y_pred))
    #print(y_pred)
    return sigmoid_balanced_cross_entropy_with_logits(logits=y_pred, labels=y_true, beta=beta)


print(balanced_binary_crossentropy(y_true, y_pred, from_logits=False))
#print(tf.reduce_mean(K.binary_crossentropy(y_true, y_pred, from_logits=False)))
print(K.binary_crossentropy(y_true, y_pred, from_logits=False))
print(balanced_cross_entropy(y_true, y_pred))
