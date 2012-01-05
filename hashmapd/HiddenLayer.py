import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T
from utils import tiled_array_image
from logistic_sgd import LogisticRegression

class HiddenLayer(object):
    def __init__(self, n_in, n_out, input, rng, poisson_layer=False, mean_doc_size=1,
            init_W=None, init_b=None, activation = T.tanh, mirroring=False):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh
        
        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.matrix/dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden 
                              layer
        """
        self.n_in = n_in
        self.n_out = n_out
        self.input = input
        self.mirroring = mirroring
        T.pprint(self.input)
        if not input:
            raise Exception
            #self.input = T.matrix('input')

        if (init_W == None):
            W_values = numpy.asarray( rng.uniform(
                    low  = -numpy.sqrt(6./(n_in+n_out)),
                    high = numpy.sqrt(6./(n_in+n_out)),
                    size = (n_in, n_out)), dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            if poisson_layer == True:
               W_values *= 1/mean_doc_size;
    
            #print 'using shared weights, randomized' #init case
            self.W = theano.shared(value = W_values) #, name ='W')
            
        else:
            #print 'using shared weights, as passed in' # unroll case
            self.W = theano.shared(value = init_W) #, name ='W')
            
        if (init_b == None):
            b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
            self.b = theano.shared(value= b_values) #, name ='b')
        else:
            self.b = theano.shared(value= init_b) #, name ='b')

        self.output = activation(T.dot(self.input, self.W) + self.b)
        # parameters of the model
        self.params = [self.W, self.b]
    
    #added MKT
    def export_model(self):
        return self.params;
    
    def load_model(self, inpt_params):
        #FIXME dont quite get why
        #self.params = inpt_params doesn't work
        #.. but it doesn't set the 'value' of shared variable W/b so we have
        #.. to do this explictly MKT
        self.W.set_value(inpt_params[0].get_value())
        self.b.set_value(inpt_params[1].get_value())
        
    def export_weights_image(self, file_name):
        # Construct image from the weight matrix        
        array = self.W.get_value()
        if not self.mirroring:
            array = array.T
        image = tiled_array_image(array)
        image.save(file_name)