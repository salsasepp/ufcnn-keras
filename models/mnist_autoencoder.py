from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.layers.convolutional_transpose import Convolution2D_Transpose

"""
   modified from https://github.com/loliverhennigh/All-Convnet-Autoencoder-Example 
   An autoencoder with 2D Convolution-Transpose layer in TF
"""

batch_size = 100 # total number of elements in the X_ and Y_ (60000 train, 10000 test) arrays must be a multiple of batch_size!
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

print("Y SHAPE",Y_train.shape)


model = Sequential()
nb_filter = 32

# input  28 * 28, output 24 * 24
model.add(Convolution2D(nb_filter, 5, 5, input_shape=((1, img_rows, img_cols)), border_mode = 'valid'))
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

# input  14 * 14 * 4, output 14 * 14 * 2
model.add(Convolution2D(2, 1, 1, border_mode = 'valid'))
model.add(Activation('relu'))

# input  input 14 * 14 * 2, output 28 * 28 * 2 ?
W_shape = [2, 2, 2, 2] # (self.nb_filter, input_dim, self.filter_length, 1)
b_shape = [2]
strides = [1,1,1,1]
deconv_shape = [2,2,2,2]
deconv_shape = [batch_size, 14, 14, 2]
model.add(Convolution2D_Transpose(deconv_shape=deconv_shape,  W_shape=W_shape, b_shape=b_shape, strides=strides, padding="SAME")) # should be lower capital same

model.add(Flatten())
model.add(Dense(784))
model.add(Reshape((1,28,28)))

"""
  W_shape --- shape of the weights - should be calculated internally
   from conv_2d:  (self.nb_filter, input_dim, self.filter_length, x)
          
   [0] ... nb_filter
   [1] ... input_dim
   [2] ... filter_length
   [3] ... 1
  
  b_shape ... shape of the biases - should be calculated internally
   [0] ... nb_filter ?

  deconv_shape
       this is output_shape of TF conv2d_transpose
       [ batch_size, input_cols, input_rows, input_depth]


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

##model.add(Dropout(0.25))
##model.add(Convolution2D_Transpose(deconv_shape=[128,18,18,1], W_shape=[3,3,1,16], b_shape=[1]))
##model.add(Activation('relu'))

print(model.summary())
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

model.compile(loss='mse', optimizer='sgd')
print("Before FIT")

#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])
