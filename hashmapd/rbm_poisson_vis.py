"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs 
to those without visible-visible and hidden-hidden connections. 
"""


import numpy, time, cPickle, gzip, PIL.Image

import theano
import theano.tensor as T
import os, sys, getopt

from utils import tiled_array_image
from theano.tensor.shared_randomstreams import RandomStreams
from load_config import LoadConfig, DefaultConfig
import rbm

class RBM_Poisson(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self,  mean_doc_size, input, input_sums,
        n_visible=784, n_hidden=500, W = None, hbias = None, vbias = None, 
        numpy_rng = None, theano_rng = None):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), 
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing 
        to a shared hidden units bias vector in case RBM is part of a 
        different network

        :param vbias: None for standalone RBMs or a symbolic variable 
        pointing to a shared visible units bias
        """

        self.mean_doc_size = mean_doc_size

        self.n_visible = n_visible
        self.n_hidden  = n_hidden


        if numpy_rng is None:    
            # create a number generator 
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None : 
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if W is None : 
           # W is initialized with `initial_W` which is uniformely sampled
           # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
           # the output of uniform if converted using asarray to dtype 
           # theano.config.floatX so that the code is runable on GPU
           initial_W = numpy.asarray( numpy_rng.uniform( 
                     low = -4.*numpy.sqrt(6./(n_hidden+n_visible)), 
                     high = 4.*numpy.sqrt(6./(n_hidden+n_visible)), 
                     size = (n_visible, n_hidden)), 
                     dtype = theano.config.floatX)
           initial_W *= 1/self.mean_doc_size
           # theano shared variables for weights and biases
           W = theano.shared(value = initial_W, name = 'W')

        if hbias is None :
           # create shared variable for hidden units bias
           hbias = theano.shared(value = numpy.zeros(n_hidden, 
                               dtype = theano.config.floatX), name='hbias')

        if vbias is None :
            # create shared variable for visible units bias
            vbias = theano.shared(value =numpy.zeros(n_visible, 
                                dtype = theano.config.floatX),name='vbias')


        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input 
        self.input_sums = input_sums
        if not input:
            self.input = T.matrix('input')
            self.input_sums = T.col('input_sums')

        self.binomial_approx_val = theano.shared(value = float(100000), name = 'binomial_approx_val')

        self.W          = W
        self.hbias      = hbias
        self.vbias      = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list 
        # other than shared variables created in this function.
        self.params     = [self.W, self.hbias, self.vbias]


    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        # see http://www.cs.toronto.edu/~hinton/absps/sh.pdf
        
        # vbias term should include:
        #
        # (-(bias_v*sample_v)) + (sample_v*log(S/N_i)) + log(v!)
        #   where S = sum_k(bias_k+sum_j(value_j*weight_kj)), ie: sum of all input activations
        #         N = no_words_in_doc
        
        # if needed, possible factorial implementation in theano (no gpu support):
        #   http://deeplearning.net/software/pylearn/api/pylearn.algorithms.sandbox.cost-pysrc.html
        
        # given the the free energy is only used to update the system parameters
        # based on its gradient (?), and since the extra factorial term has a
        # gradient of zero with respect to any of the parameters, there may not 
        # be any need to concern ourselves with explicity computing the actual
        # free energy ... and according to hinton, the udpate rule is unchanged
        
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1+T.exp(wx_b)),axis = 1)
        # extra_vbias_term1 = T.dot(v_sample,T.log(T.sum(T.dot(T.exp(self.vbias),PRODUCT_j(1+T.exp(self.W))),axis = ?))-T.log(v_sums))
        # extra_vbias_term2 = T.log(FACTORIAL(v_sample))
        return -hidden_term-vbias_term

    def propup(self, vis):
        ''' This function propagates the visible units activation upwards to
        the hidden units 
        
        Note that we return also the pre-sigmoid activation of the layer. As
        it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more
        stable computational graph (see details in the reconstruction cost function)
        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype 
        # int64 by default. If we want to keep our computations in floatX 
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size = h1_mean.shape, n = 1, p = h1_mean,
                dtype = theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid, v_sums):
        '''This function propagates the hidden units activation downwards to
        the visible units
        
        Note that we return also the pre_mean_activation of the layer. As
        it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more
        stable computational graph (see details in the reconstruction cost function)
        '''
        
        # compute the poisson mean 
        
        pre_mean_activation = T.dot(hid, self.W.T) + self.vbias
        post_mean_activation = T.nnet.softmax(pre_mean_activation)*v_sums
        
        return [pre_mean_activation,post_mean_activation]

    def sample_v_given_h(self, h0_sample, v_sums):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample, v_sums)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype 
        # int64 by default. If we want to keep our computations in floatX 
        # for the GPU we need to specify to return the dtype floatX
        
        # take a poisson sample of v1_mean (we approximate this using a binomial distribution)
        #   (note the value of n is fairly irrelevant, but larger tends to do slightly better,
        #   so we set it to be one million for now)
        
        v1_sample = self.theano_rng.binomial(size = v1_mean.shape, n = self.binomial_approx_val, p = v1_mean/self.binomial_approx_val,
                dtype = theano.config.floatX)
        
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample, v_sums):
        ''' This function implements one step of Gibbs sampling, 
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample,v_sums)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]
 
    def gibbs_vhv(self, v0_sample, v_sums):
        ''' This function implements one step of Gibbs sampling, 
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample,v_sums)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]
 
    def get_cost_updates(self, lr = 0.1, persistent=None, k=1):
        """ 
        This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM 

        :param persistent: None for CD. For PCD, shared variable containing old state
        of Gibbs chain. This must be a shared variable of size (batch size, number of
        hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The 
        dictionary contains the update rules for weights and biases but 
        also an update of the shared variable used to store the persistent
        chain, if one is used.
        """
        
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        
        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the 
        # function that implements one gibbs step k times. 
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        [pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh, 
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the 
                    # 6th output
                    outputs_info = [None,None,None,None,None,chain_start],
                    non_sequences = self.input_sums,
                    n_steps = k)
        
        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling 
        gparams = T.grad(cost, self.params,consider_constant = [chain_end])
        
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            
            # we also need to divide the learning rate by the average input size
            # as the visible values are N times larger than the [0-1] inputs into
            # the other layers
            if param.name == 'W':
                updates[param] = param - gparam * T.cast(lr/(5*self.mean_doc_size), dtype = theano.config.floatX)
            elif param.name == 'vbias':
                # vbiases don't make sense?
                updates[param] = param - gparam * T.cast(lr*0, dtype = theano.config.floatX)
            else:
                updates[param] = param - gparam * T.cast(lr, dtype = theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""
        """  (this probably doesn't work properly for the values used in poisson layer) """
        
        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name = 'bit_i_idx')
        
        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)
        
        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)
        
        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx]
        xi_flip = T.set_subtensor(xi[:,bit_i_idx], 1-xi[:, bit_i_idx])
        
        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)
        
        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i}))) 
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        
        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        
        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error
        
        Note that this function requires the pre-sigmoid activation as input. To
        understand why this is so you need to understand a bit about how
        Theano works. Whenever you compile a Theano function, the computational
        graph that you pass as input gets optimized for speed and stability. This
        is done by changing several parts of the subgraphs with others. One 
        such optimization expresses terms of the form log(sigmoid(x)) in terms of softplus. 
        We need this optimization for the cross-entropy since sigmoid of 
        numbers larger than 30. (or even less then that) turn to 1. and numbers 
        smaller than  -30. turn to 0 which in terms will force theano 
        to compute log(0) and therefore we will get either -inf or NaN 
        as cost. If the value is expressed in terms of softplus we do 
        not get this undesirable behaviour. This optimization usually works
        fine, but here we have a special case. The sigmoid is applied inside
        the scan op, while the log is outside. Therefore Theano will only 
        see log(scan(..)) instead of log(sigmoid(..)) and will not apply
        the wanted optimization. We can not go and replace the sigmoid 
        in scan with something else also, because this only needs to be 
        done on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of scan, 
        and apply both the log and sigmoid outside scan such that Theano
        can catch and optimize the expression.
        """

        cross_entropy = T.mean(T.sum((self.input/self.input_sums)*T.log(T.nnet.softmax(pre_sigmoid_nv)), axis = 1))
        return cross_entropy

## test the poisson layer by pre-training on a dataset, and printing out samples
#def test_rbm(argv = sys.argv):
#    opts, args = getopt.getopt(argv[1:], "h", ["help"])
#    
#    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])
#    
#    info = load_data_info(cfg.input.train_data_info)
#    training_prefix = info[0]
#    n_training_files = info[1]
#    n_training_batches = 10
#    dataset_postfix = '.pkl.gz'
#    batch_size = cfg.train.train_batch_size
#    mean_doc_size = info[10]
#    
#    training_data = load_poisson_data(training_prefix+'0'+dataset_postfix)
#    
#    train_data = training_data[0]
#    train_data_sums = training_data[1]
#    mean_doc_size = training_data[2]
#    
#    numpy_rng = numpy.random.RandomState(123)
#    theano_rng = RandomStreams(numpy_rng.randint(2**30))
#    
#    index  = T.lscalar()    # index to a [mini]batch
#    x      = T.matrix('input')  # the data is presented as rasterized images
#    x_sums = T.col('input_sums')
#    
#    p_rbm = RBM_Poisson(numpy_rng = numpy_rng,
#            theano_rng = theano_rng, 
#            mean_doc_size = mean_doc_size,
#            input = x,
#            input_sums = x_sums,
#            n_visible = cfg.shape.input_vector_length, 
#            n_hidden  = 500)
#    
#    # get the cost and the gradient corresponding to CD
#    cost, updates = p_rbm.get_cost_updates(lr=0.1, persistent=None, k = 1)
#    
#    #################################
#    #     Training the RBM          #
#    #################################
#    
#    print 'compiling pretraining function'
#    
#    # it is ok for a theano function to have no output
#    # the purpose of train_rbm is solely to update the RBM parameters
#    train_rbm = theano.function(inputs = [index],
#        outputs = cost,
#        updates = updates,
#        givens  = {x:train_data[index*batch_size:(index+1)*batch_size,:],
#                   x_sums:train_data_sums[index*batch_size:(index+1)*batch_size,:]})
#    
#    print 'pretraining'
#    
#    start_time = time.clock()
#    
#    # go through training epochs
#    for epoch in xrange(30):
#        
#        # go through the training set
#        mean_cost = []
#        for batch_index in xrange(n_training_batches):
#            print 'batch '+str(batch_index)+', mean weight is '+str(numpy.mean(p_rbm.W.value))+', max weight is '+str(numpy.amax(p_rbm.W.value))
#            print 'batch '+str(batch_index)+', mean vbias is '+str(numpy.mean(p_rbm.vbias.value))+', max vbias is '+str(numpy.amax(p_rbm.vbias.value))
#            print 'batch '+str(batch_index)+', mean hbias is '+str(numpy.mean(p_rbm.hbias.value))+', max hbias is '+str(numpy.amax(p_rbm.hbias.value))
#            mean_cost += [train_rbm(batch_index)]
#            print 'batch %d, cost is '%batch_index, numpy.mean(mean_cost)
#    
#        print 'Training epoch %d, cost is '%epoch, numpy.mean(mean_cost)
#    
#    end_time = time.clock()
#    pretraining_time = (end_time - start_time)
#    print ('Training took %f minutes' %(pretraining_time/60.))
#    
#    #################################
#    #     Sampling from the RBM     #
#    #################################
#
#    print 'sampling'
#
#    n_chains = 10
#    n_samples = 10
#
#    # pick random test examples, with which to initialize the persistent chain
#    test_idx = numpy_rng.randint(n_training_batches*batch_size-n_chains)
#    persistent_vis_chain = theano.shared(numpy.asarray(
#            train_data.get_value()[test_idx:test_idx+n_chains],
#            dtype=theano.config.floatX))
#    sums = theano.shared(numpy.asarray(
#            train_data_sums.get_value()[test_idx:test_idx+n_chains],
#            dtype=theano.config.floatX))
#
#    plot_every = 1000
#    # define one step of Gibbs sampling (mf = mean-field)
#    # define a function that does `plot_every` steps before returning the sample for
#    # plotting
#    [presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples], updates =  \
#                        theano.scan(p_rbm.gibbs_vhv,
#                                outputs_info = [None,None,None,None,None,persistent_vis_chain],
#                                non_sequences = sums,
#                                n_steps = plot_every)
#
#    # add to updates the shared variable that takes care of our persistent
#    # chain :.
#    updates.update({ persistent_vis_chain: vis_samples[-1]})
#    # construct the function that implements our persistent chain.
#    # we generate the "mean field" activations for plotting and the actual
#    # samples for reinitializing the state of our persistent chain
#    sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
#                                updates = updates,
#                                name = 'sample_fn')
#
#    # create a space to store the image for plotting ( we need to leave
#    # room for the tile_spacing as well)
#    image_data = numpy.zeros((29*(n_samples+1)+1,29*n_chains-1),dtype='uint8')
#    
#    # generate plot of original data (before any sampling is done)
#    image_data[0:28,:] = tile_raster_images(
#        X = (numpy.asarray(train_data.get_value()[test_idx:test_idx+n_chains],dtype=theano.config.floatX)/sums.value),
#        img_shape = (28,28),
#        tile_shape = (1, n_chains),
#        tile_spacing = (1,1))
#    
#    for idx in xrange(n_samples):
#        # generate `plot_every` intermediate samples that we discard, because successive samples in the chain are too correlated
#        vis_mf, vis_sample = sample_fn()
#        print ' ... plotting sample ', idx
#        
#        image_data[29*(idx+1):29*(idx+1)+28,:] = tile_raster_images(
#                X = (vis_mf/sums.value),
#                img_shape = (28,28),
#                tile_shape = (1, n_chains),
#                tile_spacing = (1,1))
#        # construct image
#
#    image = PIL.Image.fromarray(image_data)
#    image.save('trace/poisson_test_samples.png')
#    
#    
#    # now repeat for regular rbm
#    
#    training_data = load_data(training_prefix+'0'+dataset_postfix)
#    
#    train_data = training_data[0]
#    
#    numpy_rng = numpy.random.RandomState(123)
#    theano_rng = RandomStreams(numpy_rng.randint(2**30))
#    
#    index  = T.lscalar()    # index to a [mini]batch
#    x      = T.matrix('input')  # the data is presented as rasterized images
#    
#    b_rbm = rbm.RBM(numpy_rng = numpy_rng,
#            theano_rng = theano_rng, 
#            input = x,
#            n_visible = cfg.shape.input_vector_length, 
#            n_hidden  = 500)
#    
#    # get the cost and the gradient corresponding to CD
#    cost, updates = b_rbm.get_cost_updates(lr=0.1, persistent=None, k = 1)
#    
#    #################################
#    #     Training the RBM          #
#    #################################
#    
#    print 'compiling pretraining function'
#    
#    # it is ok for a theano function to have no output
#    # the purpose of train_rbm is solely to update the RBM parameters
#    train_rbm = theano.function(inputs = [index],
#        outputs = cost,
#        updates = updates,
#        givens  = {x:train_data[index*batch_size:(index+1)*batch_size,:]})
#    
#    print 'pretraining'
#    
#    start_time = time.clock()
#    
#    # go through training epochs
#    for epoch in xrange(30):
#        
#        # go through the training set
#        mean_cost = []
#        for batch_index in xrange(n_training_batches):
#            print 'batch '+str(batch_index)+', mean weight is '+str(numpy.mean(b_rbm.W.value))+', max weight is '+str(numpy.amax(b_rbm.W.value))
#            print 'batch '+str(batch_index)+', mean vbias is '+str(numpy.mean(b_rbm.vbias.value))+', max vbias is '+str(numpy.amax(b_rbm.vbias.value))
#            print 'batch '+str(batch_index)+', mean hbias is '+str(numpy.mean(b_rbm.hbias.value))+', max hbias is '+str(numpy.amax(b_rbm.hbias.value))
#            mean_cost += [train_rbm(batch_index)]
#            print 'batch %d, cost is '%batch_index, numpy.mean(mean_cost)
#    
#        print 'Training epoch %d, cost is '%epoch, numpy.mean(mean_cost)
#    
#    end_time = time.clock()
#    pretraining_time = (end_time - start_time)
#    print ('Training took %f minutes' %(pretraining_time/60.))
#    
#    #################################
#    #     Sampling from the RBM     #
#    #################################
#
#    print 'sampling'
#
#    n_chains = 10
#    n_samples = 10
#
#    # pick random test examples, with which to initialize the persistent chain
#    test_idx = numpy_rng.randint(n_training_batches*batch_size-n_chains)
#    persistent_vis_chain = theano.shared(numpy.asarray(
#            train_data.get_value()[test_idx:test_idx+n_chains],
#            dtype=theano.config.floatX))
#
#    plot_every = 1000
#    # define one step of Gibbs sampling (mf = mean-field)
#    # define a function that does `plot_every` steps before returning the sample for
#    # plotting
#    [presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples], updates =  \
#                        theano.scan(b_rbm.gibbs_vhv,
#                                outputs_info = [None,None,None,None,None,persistent_vis_chain],
#                                n_steps = plot_every)
#
#    # add to updates the shared variable that takes care of our persistent
#    # chain :.
#    updates.update({ persistent_vis_chain: vis_samples[-1]})
#    # construct the function that implements our persistent chain.
#    # we generate the "mean field" activations for plotting and the actual
#    # samples for reinitializing the state of our persistent chain
#    sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
#                                updates = updates,
#                                name = 'sample_fn')
#
#    # create a space to store the image for plotting ( we need to leave
#    # room for the tile_spacing as well)
#    image_data = numpy.zeros((29*(n_samples+1)+1,29*n_chains-1),dtype='uint8')
#    
#    # generate plot of original data (before any sampling is done)
#    image_data[0:28,:] = tile_raster_images(
#        X = numpy.asarray(train_data.get_value()[test_idx:test_idx+n_chains],dtype=theano.config.floatX),
#        img_shape = (28,28),
#        tile_shape = (1, n_chains),
#        tile_spacing = (1,1))
#    
#    for idx in xrange(n_samples):
#        # generate `plot_every` intermediate samples that we discard, because successive samples in the chain are too correlated
#        vis_mf, vis_sample = sample_fn()
#        print ' ... plotting sample ', idx
#        
#        image_data[29*(idx+1):29*(idx+1)+28,:] = tile_raster_images(
#                X = vis_mf,
#                img_shape = (28,28),
#                tile_shape = (1, n_chains),
#                tile_spacing = (1,1))
#        # construct image
#
#    image = PIL.Image.fromarray(image_data)
#    image.save('trace/test_samples.png')
#    

def load_data_info(info_file):
    ''' Loads info about the dataset '''
    
    print '... loading data info from ' + info_file
    
    # Load the dataset  - expecting both supervised and unsupervised data to be supplied
    f = gzip.open(info_file,'rb')
    
    training_prefix,n_training_files,n_training_batches,\
        validation_prefix,n_validation_files,n_validation_batches,\
        testing_prefix,n_testing_files,n_testing_batches,\
        batches_per_file,mean_doc_size = cPickle.load(f)
    
    f.close()
    
    return [training_prefix,n_training_files,n_training_batches,
                validation_prefix,n_validation_files,n_validation_batches,
                testing_prefix,n_testing_files,n_testing_batches,
                batches_per_file,mean_doc_size]


def load_poisson_data(dataset_file):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data from file ' + dataset_file

    # Load the dataset  - expecting both supervised and unsupervised data to be supplied (in pairs)
    f = gzip.open(dataset_file,'rb')
    x, x_sums, y = cPickle.load(f)
    f.close()

    x = x[0:100]

    x *= 255
    x = x.round()
    x_sums = x.sum(axis=1)

    mean_doc_size = int(round(numpy.mean(x_sums)))

    # shared_dataset
    shared_x = theano.shared(numpy.asarray(x,dtype=theano.config.floatX))
    
    # build a replicated 2d array of sums so operations can be performed efficiently
    shared_x_sums = theano.shared(numpy.asarray(numpy.array([x_sums]*(x.shape[1])).transpose(),dtype=theano.config.floatX))
    
    rval = [shared_x,shared_x_sums,mean_doc_size]
    return rval

def load_data(dataset_file):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data from file ' + dataset_file

    # Load the dataset  - expecting both supervised and unsupervised data to be supplied (in pairs)
    f = gzip.open(dataset_file,'rb')
    x, x_sums, y = cPickle.load(f)
    f.close()

    # shared_dataset
    shared_x = theano.shared(numpy.asarray(x,dtype=theano.config.floatX))
    
    # build a replicated 2d array of sums so operations can be performed efficiently
    shared_x_sums = theano.shared(numpy.asarray(numpy.array([x_sums]*(x.shape[1])).transpose(),dtype=theano.config.floatX))
    
    rval = [shared_x,shared_x_sums]
    return rval

if __name__ == '__main__':
    sys.exit(test_rbm())
