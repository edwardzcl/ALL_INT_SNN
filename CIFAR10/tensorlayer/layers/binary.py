# -*- coding: utf-8 -*-
import tensorflow as tf

from .. import _logging as logging
from .core import *

__all__ = [
    'Quant_DenseLayer',
    'Quant_Conv2d',
    'Quant_Layer',
    'Quant_Layer1',
]



def _quantize_median(x, k):
    G = tf.get_default_graph()
    n = float(2**k)
    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x * n + 1e-5) / n


def _quantize_weight(x, bitW, force_quantization=False):
    G = tf.get_default_graph()
    if bitW == 32 and not force_quantization:
        return x
    if bitW == 1:  # BWN
        with G.gradient_override_map({"Sign": "Identity"}):
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
            #return tf.sign(x / E) * E
            return tf.sign(x)
    #x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)  # it seems as though most weights are within -1 to 1 region anyways
    x = tf.clip_by_value(x, -1.0, 1.0)  
    return _quantize_median(x, bitW)


def _quantize_active(x, k):

    return _quantize_median(x, k)


def _cabs(x):
    #return tf.minimum(1.0, tf.abs(x), name='cabs')
    return tf.identity(x, name='cabs')


def _compute_threshold(x):
    """
    ref: https://github.com/XJTUWYD/TWN
    Computing the threshold.
    """
    x_sum = tf.reduce_sum(tf.abs(x), reduction_indices=None, keep_dims=False, name=None)
    threshold = tf.div(x_sum, tf.cast(tf.size(x), tf.float32), name=None)
    threshold = tf.multiply(0.7, threshold, name=None)
    return threshold


def _ternary_operation(x):
    """
    Ternary operation use threshold computed with weights.
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):
        threshold = _compute_threshold(x)
        x = tf.sign(tf.add(tf.sign(tf.add(x, threshold)), tf.sign(tf.add(x, -threshold))))
        return x



class Quant_DenseLayer(Layer):
    """The :class:`Quant_DenseLayer` class is a fully connected layer for quantization, in which weights are ternary {-1,0,+1} while training and inferencing.

    Note that, the bias vector would be left out.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer, usually set to ``tf.act.sign`` or apply :class:`SignLayer` after :class:`BatchNormLayer`.
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=tf.identity,
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='Quant_dense',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        logging.info("Quant_DenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args)
            # W = tl.act.sign(W)    # dont update ...
            # ternary value {-1,0,+1} for weight quantization 
            W = _ternary_operation(W)
            # W = tf.Variable(W)
            # print(W)
            if b_init is not None:
                try:
                    b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args)
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args)
                self.outputs = act(tf.matmul(self.inputs, W) + b)
                # self.outputs = act(xnor_gemm(self.inputs, W) + b) # TODO
            else:
                self.outputs = act(tf.matmul(self.inputs, W))
                # self.outputs = act(xnor_gemm(self.inputs, W)) # TODO

        self.all_layers.append(self.outputs)
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class Quant_Conv2d(Layer):
    """The :class:`Quant_Conv2d` class is a convolutional layer for quantization, in which weights are ternary {-1,0,+1} while training and inferencing.

    Note that, the bias vector would be left out.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    use_cudnn_on_gpu : bool
        Default is False.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    name : str
        A unique layer name.

    Examples
    ---------
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.Quant_Conv2d(net, 32, (5, 5), (1, 1), padding='SAME', name='bcnn1')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn1')
    ...
    >>> net = tl.layers.SignLayer(net)
    >>> net = tl.layers.Quant_Conv2d(net, 64, (5, 5), (1, 1), padding='SAME', name='bcnn2')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn2')

    """

    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=tf.identity,
            padding='SAME',
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            # act=tf.identity,
            # shape=(5, 5, 1, 100),
            # strides=(1, 1, 1, 1),
            # padding='SAME',
            # W_init=tf.truncated_normal_initializer(stddev=0.02),
            # b_init=tf.constant_initializer(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            # use_cudnn_on_gpu=None,
            # data_format=None,
            name='Quant_cnn2d',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        if act is None:
            act = tf.identity
        logging.info("Quant_Conv2d %s: n_filter:%d filter_size:%s strides:%s pad:%s act:%s" % (self.name, n_filter, str(filter_size), str(strides), padding,
                                                                                               act.__name__))

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")
        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknow input channels, set to 1")
        shape = (filter_size[0], filter_size[1], pre_channel, n_filter)
        strides = (1, strides[0], strides[1], 1)
        with tf.variable_scope(name):
            W = tf.get_variable(name='W_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args)
            # ternary value {-1,0,+1} for weight quantization 
            W = _ternary_operation(W)
            if b_init:
                b = tf.get_variable(name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args)
                self.outputs = act(
                    tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format))

        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class Quant_Layer(Layer):
    """The :class:`Quant_Layer` class is for "Median Quantization" of activations, the layer outputs to be quantized refer to quantization level and upper bound.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    k : quantization level 
    B : upper bound
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            k=1, # quantization level, default is 1
            name='sign',
    ):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        logging.info("SignLayer  %s" % (self.name))
        with tf.variable_scope(name):
            # self.outputs = tl.act.sign(self.inputs)


            #with tf.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
                #self.outputs = tf.sign(tf.add(tf.sign(tf.add(self.inputs, 0.4)), tf.sign(tf.add(self.inputs, -0.4))))
 
            #self.outputs = quantize(self.inputs)
            self.outputs = _quantize_active(_cabs(self.inputs), k)

        self.all_layers.append(self.outputs)


class Quant_Layer1(Layer):
    """The :class:`Quant_Layer` class is for "Median Quantization" of activations, the layer outputs to be quantized refer to quantization level and upper bound.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    k : quantization level 
    B : upper bound
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            k=0, # quantization level, default is 1
            B=4, # upper bound, default is 2
            name='sign',
    ):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        logging.info("SignLayer  %s" % (self.name))
        with tf.variable_scope(name):
            # self.outputs = tl.act.sign(self.inputs)


            #with tf.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
                #self.outputs = tf.sign(tf.add(tf.sign(tf.add(self.inputs, 0.4)), tf.sign(tf.add(self.inputs, -0.4))))
 
            #self.outputs = quantize(self.inputs)
            self.inputs = tf.clip_by_value(self.inputs, -B, B, name=name)
            self.outputs = _quantize_active(_cabs(self.inputs), k)

        self.all_layers.append(self.outputs)