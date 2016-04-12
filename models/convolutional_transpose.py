# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..engine import Layer, InputSpec

import tensorflow as tf


class Convolution2D_Transpose(Layer):
    """
        Creates a 2D Convolution Transpose layer (sometimes called "Deconvolution").
        Based on code by Xiaomin Wu in "[fchollet/keras] Anyone implemented a Deconvolutional                    
              layer combined the Keras and Tensorflow? (#2106)"

        Must be placed in the keras/layers/ directory.

        W_shape --- shape of the weights - should be calculated internally
         from conv_2d:  (self.nb_filter, input_dim, self.filter_length, x)

         [0] ... nb_filter
         [1] ... input_dim
         [2] ... filter_length
         [3] ... 1

        b_shape ... shape of the biases - should be calculated internally
         [0] ... nb_filter 

        deconv_shape
         this is output_shape of TF conv2d_transpose
         [ batch_size, input_cols, input_rows, input_depth]


        Input should be optimized: 
          W_shape --- shape of the weights - should be calculated internally
          b_shape ... shape of the biases - should be calculated internally

          deconv_shape of the form
              [height, width, output_channels, in_channels]
              this is set to be the Output_shape of the layer

          or output shape of the form  [None, kernel_height, kernel_width, output_depth]



          padding: VALID|SAME

          input_dim, input_length ... Keras input parameters

        Also U can set the output_shape(deconv_shape) according to:

        def conv_transpose_out_length(input_size, filter_size, border_mode, stride):
            if input_size is None:
                return None
            if border_mode == 'VALID':
                output_size = (input_size - 1) * stride + filter_size
            elif border_mode == 'SAME':
                output_size = input_size
            return output_size
    """

    input_ndim = 4

    def __init__(self,
                 init='glorot_uniform', activation='linear', weights=None,
                 padding='VALID', strides=[1,1,1,1], deconv_shape=[], W_shape = [],b_shape=[],
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, input_length=None, **kwargs):

        if padding not in {'VALID', 'SAME'}:
            raise Exception('Invalid border mode for Convolution2D:', padding)
        self.deconv_shape = deconv_shape
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert padding in {'VALID', 'SAME'}, 'border_mode must be in {VALID, SAME}'
        self.padding = padding
        self.strides = strides

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_shape = W_shape
        self.b_shape = b_shape

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        #self.input = K.placeholder(ndim=4) # old keras 0.3.x

        # Keras 1.0:
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(Convolution2D_Transpose, self).__init__(**kwargs)

    #    OLD keras 0.3.x
    #    def build(self):
    #        self.W = self.init(self.W_shape)
    #        self.b = K.zeros(self.b_shape)
    #        self.trainable_weights = [self.W, self.b]
    #        self.regularizers = []
    #
    #        if self.W_regularizer:
    #            self.W_regularizer.set_param(self.W)
    #            self.regularizers.append(self.W_regularizer)
    #
    #        if self.b_regularizer:
    #            self.b_regularizer.set_param(self.b)
    #            self.regularizers.append(self.b_regularizer)
    #
    #        if self.activity_regularizer:
    #            self.activity_regularizer.set_layer(self)
    #            self.regularizers.append(self.activity_regularizer)
    #
    #        if self.initial_weights is not None:
    #            self.set_weights(self.initial_weights)
    #            del self.initial_weights

    # NEW keras 1.0
    def build(self, input_shape):
        input_dim = input_shape[2]
        #self.W_shape = (self.nb_filter, input_dim, self.filter_length, 1) # goven from outside
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.b = K.zeros((self.b_shape), name='{}_b'.format(self.name))
        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    @property
    def output_shape(self, train = False):
        return self.deconv_shape

    def get_output_OLD(self, train=False):
        X = self.get_input(train)
        X = K.permute_dimensions(X, (0, 2, 3, 1))
        conv_out = tf.nn.conv2d_transpose(X, self.W, strides=self.strides,
                                          padding=self.padding,
                                          output_shape=self.deconv_shape)

        output = conv_out + K.reshape(self.b, (1, 1, 1, self.W_shape[2]))
        return K.permute_dimensions(output, (0, 3, 1, 2))

    def call(self, X,  mask=None):
        #X = self.get_input(train)
        X = K.permute_dimensions(X, (0, 2, 3, 1))
        conv_out = tf.nn.conv2d_transpose(X, self.W, strides=self.strides,
                                          padding=self.padding,
                                          output_shape=self.deconv_shape)

        output = conv_out + K.reshape(self.b, (1, 1, 1, self.W_shape[2]))
        return K.permute_dimensions(output, (0, 3, 1, 2))

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'padding': self.padding,
                  'strides': self.strides,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D_Transpose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_output_shape_for(self, input_shape):
        return (self.deconv_shape[0],self.deconv_shape[1],self.deconv_shape[2],self.deconv_shape[3])
        #return (None, 28, 28 ,1)
 

