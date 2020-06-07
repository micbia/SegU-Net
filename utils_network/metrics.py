from keras import backend as K
import numpy as np
import tensorflow as tf


from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops 
from tensorflow.python.ops import array_ops 
from tensorflow.python.ops import math_ops 

def sigmoid_balanced_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, beta=None, name=None):
    nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,labels, logits)
    with ops.name_scope(name, "logistic_loss", [logits, labels]) as name: 
        logits = ops.convert_to_tensor(logits, name="logits") 
        labels = ops.convert_to_tensor(labels, name="labels") 
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %(logits.get_shape(), labels.get_shape())) 
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype) 
        cond = (logits >= zeros) 
        relu_logits = array_ops.where(cond, logits, zeros) 
        neg_abs_logits = array_ops.where(cond, -logits, logits) 
        beta=0.5
        balanced_cross_entropy = relu_logits*(1.-beta)-logits*labels*(1.-beta)+math_ops.log1p(math_ops.exp(neg_abs_logits))*((1.-beta)*(1.-labels)+beta*labels)
        return tf.reduce_mean(balanced_cross_entropy)


def balanced_cross_entropy(y_true, y_pred):
    """
    To decrease the number of false negatives, set beta>1. To decrease the number of false positives, set beta<1.
    """
    beta = tf.maximum(tf.reduce_mean(1 - y_true), tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = K.log(y_pred / (1 - y_pred))
    return sigmoid_balanced_cross_entropy_with_logits(logits=y_pred, labels=y_true, beta=beta)


def tversky_loss(y_true, y_pred, beta=0.7):
    """
    Generalization of Dice’s coefficient. It adds a weight to FP (false positives) and FN (false negatives).
    For example for beta=0.5 we obtain the regular dice coefficient (see: https://arxiv.org/abs/1706.05721)
    """
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    loss = 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

    return loss



def iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    intersection = K.sum(K.abs(y_true * y_pred))
    #intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1, otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def iou_loss(y_true, y_pred):
    return 1-iou(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def phi_coef(ytrue, ypred):
    ypred_K = K.variable(np.array(ypred), dtype='float32')
    ytrue_K = K.variable(np.array(ytrue), dtype='float32')
    return K.eval(matthews_correlation(ytrue_K, ypred_K))


def focal_loss(y_true, y_pred):
    gamma, alpha = 2.0, 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
