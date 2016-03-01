from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten, Reshape, Permute, Merge, LambdaMerge, Lambda
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D, UpSampling1D, UpSampling2D
from keras.callbacks import ModelCheckpoint, Callback


def print_nodes_shapes(model):
    for k, v in model.inputs.items():
        print("{} : {} : {} : {}".format(k, type(v), v.input_shape, v.output_shape))
        
    for k, v in model.nodes.items():
        print("{} : {} : {} : {}".format(k, type(v), v.input_shape, v.output_shape))
        
    for k, v in model.outputs.items():
        print("{} : {} : {} : {}".format(k, type(v), v.input_shape, v.output_shape))


sequence_length = 5000      # same as in Roni Mittelman's paper
features = 4                # guess
nb_filter = 150             # same as in Roni Mittelman's paper
filter_length = 5           # same as in Roni Mittelman's paper
output_dim = 3              # guess

UFCNN_1 = Graph()
UFCNN_1.add_input(name='input', input_shape=(sequence_length, features))
#########################################################
UFCNN_1.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same'), name='conv1', input='input')
UFCNN_1.add_node(Activation('relu'), name='relu1', input='conv1')
#########################################################
UFCNN_1.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same'), name='conv2', input='relu1')
UFCNN_1.add_node(Activation('relu'), name='relu2', input='conv2')
#########################################################
UFCNN_1.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same'), name='conv3', input='relu2')
UFCNN_1.add_node(Activation('relu'), name='relu3', input='conv3')
#########################################################
UFCNN_1.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same'), name='conv4', input='relu3')
UFCNN_1.add_node(Activation('relu'), name='relu4', input='conv4')
#########################################################
UFCNN_1.add_node(Convolution1D(nb_filter=nb_filter,filter_length=filter_length, border_mode='same'),
                 name='conv5',
                 inputs=['relu2', 'relu4'],
                 merge_mode='sum')
UFCNN_1.add_node(Activation('relu'), name='relu5', input='conv5')
#########################################################
UFCNN_1.add_node(Convolution1D(nb_filter=nb_filter,filter_length=filter_length, border_mode='same'),
                 name='conv6',
                 inputs=['relu1', 'relu5'],
                 merge_mode='sum')
UFCNN_1.add_node(Activation('relu'), name='relu6', input='conv6')
#########################################################
UFCNN_1.add_node(Convolution1D(nb_filter=output_dim, filter_length=filter_length, border_mode='same'), name='conv7', input='relu6')
#########################################################
UFCNN_1.add_output(name='output', input='conv7')

UFCNN_1.compile(optimizer='rmsprop', loss={'output': 'categorical_crossentropy'})

print_nodes_shapes(UFCNN_1)



