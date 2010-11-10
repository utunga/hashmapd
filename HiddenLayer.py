import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T
import PIL.Image
from utils import tile_raster_images
from logistic_sgd import LogisticRegression

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, init_W=None, init_b=None, activation = T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh
        
        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden 
                              layer
        """
        self.input = input

        if (init_W == None):
            W_values = numpy.asarray( rng.uniform(
                    low  = - numpy.sqrt(6./(n_in+n_out)),
                    high = numpy.sqrt(6./(n_in+n_out)),
                    size = (n_in, n_out)), dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
    
            self.W = theano.shared(value = W_values, name ='W')
        else:
            self.W = theano.shared(value = init_W, name ='W')
        
        if (init_b == None):
            b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
            self.b = theano.shared(value= b_values, name ='b')
        else:
            self.b = theano.shared(value= init_b, name ='b')

        self.output = activation(T.dot(input, self.W) + self.b)
        # parameters of the model
        self.params = [self.W, self.b]


        #strictly for tracing only
        if (self.W.value.shape[0]==(28*28)):
            self.trace_img_shape = (28,28)
        else:
            self.trace_img_shape = (self.W.value.shape[0],1)

        if (self.W.value.shape[1]==20):
            self.trace_tile_shape = (5,4)
        else:
            self.trace_tile_shape = (10,10)
           
        self.trace_transpose_weights_file = True
        
    #added MKT
    def exportModel(self):
        return self.params;
    
    def loadModel(self, inpt_params):
        self.params=inpt_params;
        
    def export_weights_image(self, file_name):
        # Construct image from the weight matrix
        
        if (self.trace_transpose_weights_file):
            x = self.W.value.T
        else:
            x = self.W.value
            
        print 'layer '+ file_name + ' has shape '
        print self.W.value.shape       
        #print img_shape + ' ' + tile_shape
        image = PIL.Image.fromarray(tile_raster_images( x,
            img_shape = self.trace_img_shape,tile_shape = self.trace_tile_shape, 
            tile_spacing=(1,1)))
        image.save(file_name)