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
features = 1                # guess changed Ernst 20160301
nb_filter = 150             # same as in Roni Mittelman's paper
filter_length = 5           # same as in Roni Mittelman's paper
output_dim = 1              # guess changed Ernst 20160301

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

#########################################################
## Test the net with damped cosine  / remove later...
#########################################################

import matplotlib.pyplot as plt

def gen_cosine_amp(amp=100, period=25, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing
    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate

	Ernst 20160301 from https://github.com/fchollet/keras/blob/master/examples/stateful_lstm.py
        as a first test for the ufcnn
    """
    
    cos = np.zeros(((xn - x0) * step,  1, 1))
    print("Cos. Shape",cos.shape)
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(idx / (2 * np.pi * period))
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


batch_size = 64
epochs = 11 # TEST
lahead = 1

cos = gen_cosine_amp(xn = sequence_length * 10)

expected_output = np.zeros((len(cos), 1, 1))

for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    UFCNN_1.fit({'input': cos, 'output': expected_output},
              verbose=1,
              nb_epoch=1,
              shuffle=False)

print('Predicting')
predicted_output = UFCNN_1.predict({'input': cos,}, batch_size=batch_size)

print('Ploting Results')
plt.subplot(2, 1, 1)
plt.plot(expected_output.reshape(-1))
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output['output'].reshape(-1))
plt.title('Predicted')
plt.savefig('sinus.png')
#plt.show()

