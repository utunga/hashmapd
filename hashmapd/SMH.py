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
from rbm_poisson_vis import RBM_Poisson
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

    def __init__(self, numpy_rng, theano_rng = None, first_layer_type = 'bernoulli', mean_doc_size = 1, n_ins = 784, mid_layer_sizes=[200], inner_code_length = 10):
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
        
        self.first_layer_type = first_layer_type;
        self.mean_doc_size = mean_doc_size;
        
        self.sigmoid_layers = []
        self.rbm_layers     = []
        self.params         = []
        
        self.n_ins = n_ins
        self.inner_code_length = inner_code_length
        self.mid_layer_sizes = mid_layer_sizes
        
        self.numpy_rng = numpy_rng
        self.theano_rng = RandomStreams(numpy_rng.randint(2**30))
     
        # allocate symbolic variables for the data
        
        if (theano.config.floatX == "float32"):
            self.x  = T.matrix('x')  #
            self.x_sums = T.matrix('x_sums')
            self.y  = T.matrix('y') # the output (after finetuning) should look the same as the input
        else:
            if (theano.config.floatX == "float64"):
                self.x  = T.dmatrix('x')  #
                self.x_sums = T.dmatrix('x_sums')
                self.y  = T.dmatrix('y') # the output (after finetuning) should look the same as the input
            else:        
                raise Exception #not sure whats up here..

        # The SMH is an MLP, for which all weights of intermediate layers are shared with a
        # different RBM.  We will first construct the SMH as a deep multilayer perceptron, and
        # when constructing each sigmoidal layer we also construct an RBM that shares weights
        # with that layer. During pretraining we will train these RBMs (which will lead
        # to chainging the weights of the MLP as well) During finetuning we will finish
        # training the SMH by doing stochastic gradient descent on the MLP.

        self.init_layers()
    
    def init_layers(self):
        
        ### input-layer 0 (n_ins->50) *including RBM
      
        inner_code_length = self.inner_code_length
        hidden_layer_sizes = self.mid_layer_sizes + [inner_code_length]
        
        # middle layer/layers
        num_hidden = len(hidden_layer_sizes)
        for i in xrange(num_hidden):
            # the input is x if we are on the first layer, otherwise input to this layer is output of layer below
            n_out = hidden_layer_sizes[i]
            
            if i == 0 and self.first_layer_type == 'poisson':
                n_in = self.n_ins
                layer_input = self.x
                sigmoid_layer = HiddenLayer(rng   = self.numpy_rng, 
                                        input = layer_input, 
                                        poisson_layer = True,
                                        mean_doc_size = self.mean_doc_size,
                                        n_in  = n_in, 
                                        n_out = n_out,
                                        activation = T.nnet.sigmoid)
            else:
                if i == 0:
                    n_in = self.n_ins
                    layer_input = self.x
                else:
                    n_in = hidden_layer_sizes[i-1]
                    layer_input = self.sigmoid_layers[-1].output
                sigmoid_layer = HiddenLayer(rng   = self.numpy_rng, 
                                        input = layer_input, 
                                        n_in  = n_in, 
                                        n_out = n_out,
                                        activation = T.nnet.sigmoid)

            
            #print 'created layer(n_in:%d n_out:%d)'%(sigmoid_layer.n_in,sigmoid_layer.n_out)            
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            
            # Construct an RBM that shared weights with this layer (first layer is Poisson)
            if i == 0 and self.first_layer_type == 'poisson':
                rbm_layer = RBM_Poisson(numpy_rng = self.numpy_rng,
                                theano_rng = self.theano_rng, 
                                input = layer_input, 
                                input_sums = self.x_sums,
                                mean_doc_size = self.mean_doc_size,
                                n_visible = n_in, 
                                n_hidden  = n_out,
                                W = sigmoid_layer.W, #NB data is shared between the RBM and the sigmoid layer
                                hbias = sigmoid_layer.b)
            else :
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
    
        self.n_rbm_layers = len(self.rbm_layers)
        self.n_sigmoid_layers = len(self.sigmoid_layers)
    
    def unroll_layers(self, cost, noise_std_dev):
    
        inner_code_length = self.inner_code_length
        hidden_layer_sizes = self.mid_layer_sizes + [inner_code_length]
        num_hidden = len(hidden_layer_sizes)
        
        # create a new random stream for generating gaussian noise
        srng = RandomStreams(numpy.random.RandomState(234).randint(2**30))
        
        for i in xrange(num_hidden):
            reverse_indx = num_hidden-i-1 
            mirror_layer = self.sigmoid_layers[reverse_indx];
            
            # add gaussian noise to codes (the middle layer) for fine tuning
            if i == 0 and noise_std_dev > 0:
                layer_input = self.sigmoid_layers[-1].output+srng.normal(self.sigmoid_layers[-1].output.shape,avg=0.0,std=noise_std_dev);
            else:
                layer_input = self.sigmoid_layers[-1].output
            
            # create the relevant layer (last layer is a softmax layer which we calculate the cross entropy error of during fine tuning)
            if i == num_hidden and cost == 'cross_entropy':
                self.logRegressionLayer = HiddenLayer(
                                  input = layer_input,
                                  n_in  = mirror_layer.n_out,
                                  n_out = mirror_layer.n_in,
                                  init_W = mirror_layer.W.value.T,
                                  activation = T.nnet.softmax)
            else:
                sigmoid_layer = HiddenLayer(rng   = self.numpy_rng, 
                                        input = layer_input, 
                                        n_in  = mirror_layer.n_out, 
                                        n_out = mirror_layer.n_in,
                                        init_W = mirror_layer.W.value.T,
                                        #init_b = mirror_layer.b.value.reshape(mirror_layer.b.value.shape[0],1), #cant for the life of me think of a good default for this 
                                        activation = T.nnet.sigmoid)
            
            #print 'created layer(n_in:%d n_out:%d)'%(sigmoid_layer.n_in,sigmoid_layer.n_out)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params) ##NB NN training gradients are computed with respect to self.params

        self.y = self.sigmoid_layers[-1].output; 
        self.n_sigmoid_layers = len(self.sigmoid_layers)
        
        # compute the cost (cross entropy) for second phase of training
        # can't get nll to work so just use squared diff to get something working! (MKT)
        if cost == 'cross_entropy':
            self.finetune_cost = self.cross_entropy_error()
        else:
            self.finetune_cost = self.squared_diff_cost()
    
    #static versions of above - useful for debugging
    #def static_init_layers(self):
    #    ###    MKT: desperately dbugging, back to static wiring up again
    #    
    #    ### input-layer 0 (n_ins->2)
    #    layer_input = self.x
    #    sigmoid_layer = HiddenLayer(rng   = self.numpy_rng, 
    #                                input = layer_input, 
    #                                n_in  = self.n_ins, 
    #                                n_out = self.inner_code_length,
    #                                activation = T.nnet.sigmoid)
    #    print 'created layer(n_in:%d n_out:%d)'%(sigmoid_layer.n_in,sigmoid_layer.n_out)            
    #    self.sigmoid_layers.append(sigmoid_layer)
    #    self.params.extend(sigmoid_layer.params)
    #    self.hash_code_layer = self.sigmoid_layers[-1]
    #
    #    # Construct an RBM that shared weights with this layer
    #    rbm_layer = RBM(numpy_rng = self.numpy_rng,
    #                    theano_rng = self.theano_rng, 
    #                    input = layer_input, 
    #                    n_visible = sigmoid_layer.n_in, 
    #                    n_hidden  = sigmoid_layer.n_out,
    #                    W = sigmoid_layer.W, #NB data is shared between the RBM and the sigmoid layer
    #                    hbias = sigmoid_layer.b)
    #    print 'created rbm (n_visible:%d n_hidden:%d)'%(rbm_layer.n_visible,rbm_layer.n_hidden)                  
    #    self.rbm_layers.append(rbm_layer)
    #    self.n_rbm_layers = len(self.rbm_layers)
    #    self.n_sigmoid_layers = len(self.sigmoid_layers)
    #   
    #    self.unroll_layers = self.static_unroll_layers
    #    
    #def static_unroll_layers(self):
    #    print 'unrolling layers'
    #    mirror_layer = self.sigmoid_layers[-1];
    #    layer_input = self.sigmoid_layers[-1].output
    #    
    #    #print 'mirror_layer : (',mirror_layer.n_in,',', mirror_layer.n_out,')'
    #    #print 'mirror_layer.b.shape :', mirror_layer.b.value.shape - prints, eg fake_img mirror_layer.b.shape : (4,)
    #    sigmoid_layer = HiddenLayer(rng   = self.numpy_rng, 
    #                                input = layer_input, 
    #                                n_in  = mirror_layer.n_out, 
    #                                n_out = mirror_layer.n_in,
    #                                init_W = mirror_layer.W.value.T,
    #                                #init_b = mirror_layer.b.value,
    #                                activation = T.nnet.sigmoid)
    #    
    #    #print 'sigmoid_layer.b.shape :', sigmoid_layer.b.value.shape prints, eg fake_img sigmoid_layer.b.shape : (784,)
    #
    #    print 'created layer(n_in:%d n_out:%d)'%(sigmoid_layer.n_in,sigmoid_layer.n_out)
    #    self.sigmoid_layers.append(sigmoid_layer)
    #    self.params.extend(sigmoid_layer.params) #NB NN training gradients are computed with respect to self.params
    #
    #    self.y = self.sigmoid_layers[-1].output;
    #    
    #    self.n_sigmoid_layers = len(self.sigmoid_layers)
    #    
    #    # compute the cost for second phase of training
    #    # can't get nll to work so just use squared diff to get something working! (MKT)
    #    self.finetune_cost = self.squared_diff_cost() #negative_log_likelihood()
    #    
    def output_given_x(self, data_x):
        
        output_fn = theano.function( [],
               outputs =  self.sigmoid_layers[-1].output, 
               givens  = {self.sigmoid_layers[0].input : data_x})
        
        return output_fn();

    def output_codes_given_x(self, data_x):
        #print theano.pprint(self.x), len(data_x), data_x[0].shape, data_x[1].shape, data_x[2].shape 
        output_fn = theano.function( [],
               outputs =  self.hash_code_layer.output, 
               givens  = {self.x : data_x})
        
        return output_fn();

    def pretraining_functions(self, batch_size, method, k):
        ''' Generates a list of functions, for performing one step of gradient descent at a
        given layer. The function will require as input a minibatch of data, and to train an
        RBM you just need to iterate, calling the corresponding function on all minibatches.
        
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :type method: string
        :param method: type of Gibbs sampling to perform: 'cd' (default) or 'pcd'
        :type k: int
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        '''
        
        learning_rate = T.scalar('lr')    # learning rate to use
        train_set_x = T.matrix('train_set_x')
        train_set_x_sums = T.matrix('train_set_x_sums')
        
        pretrain_fns = []
        for rbm in self.rbm_layers:
            if method == 'pcd':
                # initialize storage for the persistent chain (state = hidden layer of chain)
                persistent_chain = theano.shared(numpy.zeros((batch_size,rbm.n_hidden),dtype=theano.config.floatX))
                # get the cost and the gradient corresponding to one step of PCD-k
                cost,updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=k)
            else:
                # default = use CD instead
                cost,updates = rbm.get_cost_updates(lr=learning_rate, persistent=None, k=k)
            
            # compile the theano function    
            fn = theano.function(inputs = [train_set_x,train_set_x_sums,
                        theano.Param(learning_rate, default = 0.1)],
                    outputs = cost,
                    updates = updates,
                    givens  = {self.x:train_set_x,
                               self.x_sums:train_set_x_sums}
                    # uncomment the following line to perform debugging:
                    #   ,mode=theano.compile.debugmode.DebugMode(stability_patience=5)
                    )
            
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    # compute the cross entropy error between the data and its reconstruction
    #     NB: this is the multiple classes (softmax) form - as per bishop
    def cross_entropy_error(self):
        # infer reconstructed data
        recon = self.sigmoid_layers[-1].output
        # convert intput data to probabilities if appropriate
        x = self.x
        if self.first_layer_type == 'poisson':
            x = x/self.x_sums;
        # determine error (see note below about potentially the using mean instead of the sum)
        cross_entropy = (x*T.log(recon)).sum(axis = 1)
        return -cross_entropy.mean()

    # compute the mean squared error between the data and its reconstruction
    def squared_diff_cost(self):
        # infer reconstructed data, and scale output up to match input if appropriate
        recon = self.sigmoid_layers[-1].output
        # convert intput data to probabilities if appropriate
        x = self.x
        if self.first_layer_type == 'poisson':
            x = x/self.x_sums;
        # determine error
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

    def build_finetune_functions(self, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of finetuning, a function
        `validate` that computes the error on a batch from the validation set, and a function
        `test` that computes the error on a batch from the testing set
        
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''
        
        train_set_x = T.matrix('train_set_x')
        train_set_x_sums = T.matrix('train_set_x_sums')
        valid_set_x = T.matrix('valid_set_x')
        valid_set_x_sums = T.matrix('valid_set_x_sums')
        test_set_x = T.matrix('test_set_x')
        test_set_x_sums = T.matrix('test_set_x_sums')
        
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)
        
        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam*learning_rate
        
        train_fn = theano.function(inputs = [train_set_x, train_set_x_sums], 
              outputs =  self.finetune_cost, 
              updates = updates,
              givens  = { self.x : train_set_x,
                          self.x_sums : train_set_x_sums })
        
        valid_score_i = theano.function([valid_set_x, valid_set_x_sums], self.finetune_cost,
              givens  = { self.x : valid_set_x,
                          self.x_sums : valid_set_x_sums })
        
        test_score_i = theano.function([test_set_x, test_set_x_sums], self.finetune_cost,
              givens  = { self.x : test_set_x,
                          self.x_sums : test_set_x_sums })
        
        return train_fn, valid_score_i, test_score_i
    
    
    #added MKT
    def export_model(self):
        joint_params = []
        for layer in self.sigmoid_layers:
            joint_params.append(layer.export_model())
        return joint_params;
    
    def load_model(self, inpt_params):
        for i in xrange(len(inpt_params)):
            #print 'loading layer %i'%i
            self.sigmoid_layers[i].load_model(inpt_params[i])


