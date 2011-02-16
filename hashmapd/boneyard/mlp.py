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

    def export_model():
        return self.params;
    
    def load_model(inpt_params):
        self.params=inpt_params;




