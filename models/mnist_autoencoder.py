from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.layers.convolutional_transpose import Convolution2D_Transpose

"""
   modified from https://github.com/loliverhennigh/All-Convnet-Autoencoder-Example 
   An autoencoder with 2D Convolution-Transpose layer in TF
"""

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 28 * 14
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 2

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_train = X_train
Y_test  = X_test


model = Sequential()
nb_filter = 6464
nb_depth =  10

# input  28 * 28, output 24 * 24
model.add(Convolution2D(nb_filter, 5, 5, input_shape=(1, img_rows, img_cols), border_mode = 'valid'))
model.add(Activation('relu'))

# input  24 * 24, output 20 * 20
model.add(Convolution2D(nb_filter, 5, 5, border_mode = 'valid'))
model.add(Activation('relu'))

# input  20 * 20, output 16 * 16
model.add(Convolution2D(nb_filter, 5, 5, border_mode = 'valid'))
model.add(Activation('relu'))

# input  16 * 16, output 14 * 14
model.add(Convolution2D(nb_filter, 3, 3, border_mode = 'valid'))
model.add(Activation('relu'))


model.add(Convolution2D_Transpose(deconv_shape=[2,2,nb_filter,nb_filter],  W_shape=[2,2,nb_filter,nb_filter], b_shape=[nb_filter], strides=[1,1,1,1], padding="SAME"))


"""
  W_shape --- shape of the weights - should be calculated internally
  b_shape ... shape of the biases - should be calculated internally

  deconv_shape of the form  [kernel_height, kernel_width, output_depth, input_depth]

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
model.add(Activation('relu'))

##model.add(Dropout(0.25))
##model.add(Convolution2D_Transpose(deconv_shape=[128,18,18,1], W_shape=[3,3,1,16], b_shape=[1]))
##model.add(Activation('relu'))


#  OLD MNIST MODEL
#model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
#                        border_mode='valid',
#                        input_shape=(1, img_rows, img_cols)))
#model.add(Activation('relu'))
#model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
