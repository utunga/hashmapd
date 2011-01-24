"""
"""
import os

import numpy, time, cPickle, gzip, os, sys

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from HiddenLayer import HiddenLayer
from rbm import RBM
from logistic_sgd import LogisticRegression


class SMH(object):
    """semantic hasher - based on the SMH code
    
    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the 
    network, and the hidden layer of the last RBM represents the output.
    
    When fine tuning we train via minimizing cross entropy of difference between
    inputs and inputs after squashing to top layer and back.
    
    """

    def __init__(self, numpy_rng, theano_rng = None, n_ins = 784, mid_layer_sizes=[200], inner_code_length = 10):
        """This class is made to support a variable number of layers. 

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial 
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is 
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input (and autoencoder output, y) of the SMH 

        :type n_code_length: int
        :param n_code_length: how many codes to squash down to in the middle layer
        """
        
        self.sigmoid_layers = []
        self.rbm_layers     = []
        self.params         = []
        
        self.n_ins = n_ins
        self.inner_code_length = inner_code_length
        self.mid_layer_sizes = mid_layer_sizes
        
        self.numpy_rng = numpy_rng
        self.theano_rng = RandomStreams(numpy_rng.randint(2**30))
     
        # allocate symbolic variables for the data
        self.x  = T.dmatrix('x')  # the data is presented as rasterized images
        self.y  = T.dmatrix('y') # the output (after finetuning) should look the same as the input

        # The SMH is an MLP, for which all weights of intermediate layers are shared with a
        # different RBM.  We will first construct the SMH as a deep multilayer perceptron, and
        # when constructing each sigmoidal layer we also construct an RBM that shares weights
        # with that layer. During pretraining we will train these RBMs (which will lead
        # to chainging the weights of the MLP as well) During finetuning we will finish
        # training the SMH by doing stochastic gradient descent on the MLP.

        self.init_test_layers()
    
        # compute the cost for second phase of training
        # can't get nll to work so just use squared diff to get something working! (MKT)
        self.finetune_cost = self.squared_diff_cost() #negative_log_likelihood()

    def init_test_layers(self):
        ###    MKT: To keep things simple for now, we hard code the layers structure right in this constructor
        
        ### input-layer 0 (n_ins->50) *including RBM
      
        inner_code_length = self.inner_code_length
        hidden_layer_sizes = self.mid_layer_sizes + [inner_code_length]
                
        # middle layer/layers
        num_hidden = len(hidden_layer_sizes)
        for i in xrange(num_hidden):
            # the input is x if we are on the first layer, otherwise input to this layer is output of layer below
            if i == 0 :
                n_in = self.n_ins
                layer_input = self.x
            else:
                n_in = hidden_layer_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output

            n_out = hidden_layer_sizes[i]

            sigmoid_layer = HiddenLayer(rng   = self.numpy_rng, 
                                        input = layer_input, 
                                        n_in  = n_in, 
                                        n_out = n_out,
                                        activation = T.nnet.sigmoid)
            print 'created layer(n_in:%d n_out:%d)'%(sigmoid_layer.n_in,sigmoid_layer.n_out)            
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
        
            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng = self.numpy_rng,
                            theano_rng = self.theano_rng, 
                            input = layer_input, 
                            n_visible = n_in, 
                            n_hidden  = n_out,
                            W = sigmoid_layer.W, #NB data is shared between the RBM and the sigmoid layer
                            hbias = sigmoid_layer.b)
            #print 'created rbm (n_in:%d n_out:%d)'%(rbm_layer.n_in,rbm_layer.n_out)                  
            self.rbm_layers.append(rbm_layer)
        
        self.hash_code_layer = self.sigmoid_layers[-1]
    
        ###  layer 2 (hashcode) - layer 3 (n_code_length->mid_layer_size)
        for i in xrange(num_hidden):
            rev = num_hidden-i-1 #yeah there must be a nicer way to do this but I dont know python and on train right now sorry
        
            mirror_layer = self.sigmoid_layers[rev];
            layer_input = self.sigmoid_layers[-1].output
            sigmoid_layer = HiddenLayer(rng   = self.numpy_rng, 
                                        input = self.sigmoid_layers[-1].output, 
                                        n_in  = mirror_layer.n_out, 
                                        n_out = mirror_layer.n_in,
                                        init_W = mirror_layer.W.value.T,
                                        #init_b = mirror_layer.b.value.reshape(mirror_layer.b.value.shape[0],1),
                                        activation = T.nnet.sigmoid)
            print 'created layer(n_in:%d n_out:%d)'%(sigmoid_layer.n_in,sigmoid_layer.n_out)
            self.sigmoid_layers.append(sigmoid_layer)
            #self.params.extend(sigmoid_layer.params)

        self.y = self.sigmoid_layers[-1].output;
        
        self.n_rbm_layers = num_hidden
        self.n_unrolled_layers = len(self.sigmoid_layers)
        
        
    def output_given_x(self, data_x):
        
        output_fn = theano.function( [],
               outputs =  self.sigmoid_layers[-1].output, 
               givens  = {self.sigmoid_layers[0].input : data_x})
        
        return output_fn();

    def output_codes_given_x(self, data_x):
        
        output_fn = theano.function( [],
               outputs =  self.hash_code_layer.output, 
               givens  = {self.sigmoid_layers[0].input : data_x})
        
        return output_fn();

    def pretraining_functions(self, train_set_x, batch_size,k):
        ''' Generates a list of functions, for performing one step of gradient descent at a
        given layer. The function will require as input the minibatch index, and to train an
        RBM you just need to iterate, calling the corresponding function on all minibatch
        indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        '''

        # index to a [mini]batch
        index            = T.lscalar('index')   # index to a minibatch
        learning_rate    = T.scalar('lr')    # learning rate to use

        # number of batches
        n_batches = train_set_x.value.shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin+batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost,updates = rbm.get_cost_updates(learning_rate, persistent=None, k =k)

            # compile the theano function    
            fn = theano.function(inputs = [index, 
                              theano.Param(learning_rate, default = 0.1)],
                    outputs = cost, 
                    updates = updates,
                    givens  = {self.x :train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def squared_diff_cost(self):
        recon = self.sigmoid_layers[-1].output
        x = self.x
        squared_diff = (x-recon)**2
        return squared_diff.mean()
        
    #cant get this to work right
    #def negative_log_likelihood(self):
    #    """Return the mean of the negative log-likelihood of the prediction
    #    of this model under a given target distribution.
    #
    #    Note: we use the mean instead of the sum so that
    #          the learning rate is less dependent on the batch size
    #    """
    #    return -T.mean(T.mean(self.x*T.log(self.sigmoid_layers[-1].output) + (1 - self.x)*T.log(self.sigmoid_layers[-1].output), axis = 1), axis=0)
    #   
    #def errors(self):
    #    return T.mean(self.negative_log_likelihood(), axis=0)

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of finetuning, a function
        `validate` that computes the error on a batch from the validation set, and a function
        `test` that computes the error on a batch from the testing set
    
        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;  the has to contain three
        pairs, `train`, `valid`, `test` in this order, where each pair is formed of two Theano
        variables, one for the datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''
        
        train_set_x = datasets[0]
        valid_set_x = datasets[1]
        test_set_x = datasets[2]
    
        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.value.shape[0] / batch_size
        n_test_batches  = test_set_x.value.shape[0]  / batch_size
    
        index   = T.lscalar('index')    # index to a [mini]batch 
    
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)
    
        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam*learning_rate
    
        train_fn = theano.function(inputs = [index], 
              outputs =  self.finetune_cost, 
              updates = updates,
              givens  = { self.x : train_set_x[index*batch_size:(index+1)*batch_size]})
    
        test_score_i = theano.function([index], self.finetune_cost,
                 givens = {
                   self.x: test_set_x[index*batch_size:(index+1)*batch_size]})
    
        valid_score_i = theano.function([index], self.finetune_cost,
              givens = {
                 self.x: valid_set_x[index*batch_size:(index+1)*batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]
    
        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]
    
        return train_fn, valid_score, test_score

    #added MKT
    def exportModel(self):
        joint_params = []
        for layer in self.sigmoid_layers:
            joint_params.append(layer.exportModel())
        return joint_params;
    
    def loadModel(self, inpt_params):
        for i in xrange(len(inpt_params)):
            #print 'loading layer %i'%i
            self.sigmoid_layers[i].loadModel(inpt_params[i])


