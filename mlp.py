"""
This tutorial introduces the multilayer perceptron using Theano.  

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear 
activation function (usually tanh or sigmoid) . One can use many such 
hidden layers making the architecture deep. The tutorial will also tackle 
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" - 
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation = T.tanh):
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

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you 
        #        should use 4 times larger initial weights for sigmoid 
        #        compared to tanh
        #        We have no info for other function, so we use the same as tanh.
        W_values = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_in+n_out)),
                high = numpy.sqrt(6./(n_in+n_out)),
                size = (n_in, n_out)), dtype = theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value = W_values, name ='W')

        b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
        self.b = theano.shared(value= b_values, name ='b')

        self.output = activation(T.dot(input, self.W) + self.b)
        # parameters of the model
        self.params = [self.W, self.b]

    #added MKT
    def exportModel():
        return self.params;
    
    def loadModel(inpt_params):
        self.params=inpt_params;

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model 
    that has one layer or more of hidden units and nonlinear activations. 
    Intermediate layers usually have as activation function thanh or the 
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the 
    top layer is a softamx layer (defined here by a ``LogisticRegression`` 
    class). 
    """



    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the 
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in 
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units 

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in 
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will 
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer 
        # implementing any other nonlinearity
        self.hiddenLayer = HiddenLayer(rng = rng, input = input, 
                                 n_in = n_in, n_out = n_hidden,
                                 activation = T.tanh)

        # The logistic regression layer gets as input the hidden units 
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression( 
                                    input = self.hiddenLayer.output,
                                    n_in  = n_hidden,
                                    n_out = n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to 
        # be small 
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce 
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W**2).sum() \
                    + (self.logRegressionLayer.W**2).sum()

        # negative log likelihood of the MLP is given by the negative 
        # log likelihood of the output of the model, computed in the 
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

    def exportModel():
        return self.params;
    
    def loadModel(inpt_params):
        self.params=inpt_params;




