import os
import sys

from keras import backend as K
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, Graph, Model
from keras.models import model_from_json

from keras.layers import Input, merge, Flatten, Dense, Activation, Convolution1D, ZeroPadding1D
from keras.layers import TimeDistributed, Reshape
from keras.layers.recurrent import LSTM

from keras.engine import training 


class Models(object):

    def __init__(self):
        pass

    def save_model (self, model, model_name):
        locpath="./"
        json_string = model.to_json()
        open(locpath + model_name + '_architecture.json', 'w').write(json_string)
        model.save_weights(locpath + model_name + '_weights.h5', overwrite=True)

        yaml_string = model.to_yaml()
        with open(locpath + model_name + '_data.yml', 'w') as outfile:
            outfile.write( yaml_string)

    def load_model(self, model_name):
        """ 
        reading the model from disk - including all the trained weights and the complete model design (hyperparams, planes,..)
        """
    
        locpath="./"
        arch_name = locpath + model_name + '_architecture.json'
        weight_name = locpath + model_name + '_weights.h5'
    
        if not os.path.isfile(arch_name) or not os.path.isfile(weight_name):
            print("model_name given and file %s and/or %s not existing. Aborting." % (arch_name, weight_name))
            sys.exit()

        print("Loaded model: ",model_name)

        try:
            model = model_from_json(open(arch_name).read(),{'Convolution1D_Transpose_Arbitrary':Convolution1D_Transpose_Arbitrary})
        except NameError:
            model = model_from_json(open(arch_name).read())

        model.load_weights(weight_name)
        self.model = model
        return model


    def model_ufcnn_concat(self, sequence_length=5000,
                       features=1,
                       nb_filter=150,
                       filter_length=5,
                       output_dim=1,
                       optimizer='adagrad',
                       loss='mse',
                       batch_size = 512,
                       regression = True,
                       class_mode=None,
                       activation="softplus",
                       init="lecun_uniform"):

        main_input = Input(name='input', shape=(sequence_length, features))

        #########################################################

        #input_padding = ZeroPadding1D(2)(main_input)  # to avoid lookahead bias

        #########################################################

        conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(main_input)
        relu1 = Activation(activation)(conv1)

        #########################################################

        conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu1)
        relu2 = Activation(activation)(conv2)

        #########################################################

        conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu2)
        relu3 = Activation(activation)(conv3)

        #########################################################

        conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu3)
        relu4 = Activation(activation)(conv4)

        #########################################################

        conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu4)
        relu5 = Activation(activation)(conv5)

        #########################################################

        merge6 = merge([relu3, relu5], mode='concat')
        conv6 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(merge6)
        relu6 = Activation(activation)(conv6)

        #########################################################

        merge7 = merge([relu2, relu6], mode='concat')
        conv7 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(merge7)
        relu7 = Activation(activation)(conv7)

        #########################################################

        merge8 = merge([relu1, relu7], mode='concat')
        conv8 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(merge8)
        relu8 = Activation(activation)(conv8)

        #########################################################
        if regression:
            #########################################################

            conv9 = Convolution1D(nb_filter=output_dim, filter_length=filter_length, border_mode='same', init=init)(relu8)
            output = conv9
            #main_output = conv9.output

        else:

            conv9 = Convolution1D(nb_filter=output_dim, filter_length=sequence_length, border_mode='valid', init=init)(relu8)
            activation9 = (Activation(activation))(conv9)
            flat = Flatten () (activation9)
            # activation in the last layer should be linear... and a one dimensional array.....
            dense = Dense(output_dim)(flat)
            output = dense
    
        model = Model(input=main_input, output=output)
        model.compile(optimizer=optimizer, loss=loss)

        print(model.summary())

        self.model = model
        return model

    def atari_conv_model(self, sequence_length=5000,
                       features=1,
                       output_dim=1,
                       optimizer='adagrad',
                       activation='softplus',
                       loss='mse',
                       batch_size = 512,
                       init="lecun_uniform"):
        """
        After the ConvModel in Deep Mind RL paper: Mnih, V., et al. "Asynchronous Methods for Deep Reinforcement Learning"
        """

        main_input = Input(name='input', shape=(sequence_length, features))

        #########################################################

        #input_padding = ZeroPadding1D(2)(main_input)  # to avoid lookahead bias

        #########################################################

        conv1 = Convolution1D(nb_filter=16, filter_length=8, border_mode='valid', init=init, subsample_length=4)(main_input)
        relu1 = Activation(activation)(conv1)

        #########################################################

        conv2 = Convolution1D(nb_filter=32, filter_length=4, border_mode='valid', subsample_length=2, init=init)(relu1)
        relu2 = Activation(activation)(conv2)

        #########################################################

        flat = Flatten () (relu2)
        dense1 = Dense(256)(flat)
        relu3 = Activation(activation)(dense1)

        # activation in the last layer should be linear... and a one dimensional array.....
        dense2 = Dense(output_dim)(relu3)
        output = dense2
    
        model = Model(input=main_input, output=output)
        model.compile(optimizer=optimizer, loss=loss)

        print(model.summary())
        self.model = model
        return model


    def get_layer_weights(self):
        """
        get the layers of the current model...
        """
        result = {} 
        jlayer=0
        for layer in self.model.layers:
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            weight_values = K.batch_get_value(symbolic_weights)
            layer_list=[]
            for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                layer_list.append(val)
            print("ADDING W",jlayer)
            result[jlayer] = layer_list
            jlayer += 1
        return result 


    def set_layer_weights(self, input_weights):
        """
        set the weights of the current model to the input weights...
        """
        #for name, val in results:
        jlayer=0
        weight_value_tuples = []
        for layer in self.model.layers:
            weight_values = input_weights[jlayer]
            jlayer += 1
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            weight_value_tuples += zip(symbolic_weights, weight_values)

        K.batch_set_value(weight_value_tuples)

    def get_gradients_numeric(self, x, y, updates=[]):
        cost_s, grads_s = self.get_cost_grads_symbolic()
        #sample_weights missing...

        #inputs = x + y + self.sample_weights + [1.]
        ## x and y must come from Keras ans are placeholdes
        inputs = [x] + [y] + [] + [1.]
        train_function = K.function(inputs,
                                    [grads_s],
                                     updates=updates)


        f = train_function
        outs = f(inputs)
        print (outs)


     
    def get_cost_grads_symbolic(self):
        """ Returns symbolic cost and symbolic gradients for the model """
        trainable_params = self._get_trainable_params(self.model)

        #cost = self.model.model.total_loss
        cost = self.model.total_loss
        grads = K.gradients(cost, trainable_params)

        return cost, grads


    def _get_trainable_params(self, model):
        params = []
        for layer in model.layers:
            params += training.collect_trainable_weights(layer)
        return params

    def get_training_function(self, x,y):
        # get trainable weights
        trainable_weights = []
        for layer in self.model.layers:
            trainable_weights += collect_trainable_weights(layer)

        # get the grads - more or less
        weights = [K.variable(np.zeros(K.get_value(p).shape)) for p in trainable_weights]
        training_updates = self.optimizer.get_updates(trainable_weights, self.constraints, self.total_loss)




