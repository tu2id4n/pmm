import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    print(scaled_images)
    # scaled_images = tf.transpose(scaled_images, [0, 3, 1, 2])
    # print(scaled_images)
    layer_1 = activ(
        conv(scaled_images, 'c1', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    print('layer_1', layer_1)
    layer_2 = activ(
        conv(layer_1, 'c2', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    print('layer_2', layer_2)
    layer_3 = activ(
        conv(layer_2, 'c3', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    print('layer_3', layer_3)
    # layer_4 = activ(
    #     conv(layer_3, 'c4', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    # layer_5 = activ(
    #     conv(layer_4, 'c5', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), pad='SAME', **kwargs))
    layer_5 = conv_to_fc(layer_3)

    return activ(linear(layer_5, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
