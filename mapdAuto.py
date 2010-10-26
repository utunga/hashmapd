import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T

from mlp import HiddenLayer
from dA import dA
from theano.tensor.shared_randomstreams import RandomStreams
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Some fileconstants

#real data
WEIGHTS_FILE = "data/model_params.pkl"
#PICKLED_WORD_VECTORS_FILE = "data/word_vectors.pkl.gz";
#TEST_PICKLED_WORD_VECTORS_FILE = "data/test_word_vectors.pkl.gz"
#
#WORD_VECTORS_NUM_WORDS = 3000; #number of different words in above file ('word_id' column is allowed to be *up to and including* this number)
#WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)

#test data
#WORD_VECTORS_FILE = "data/test_word_vectors.csv";
#WORD_VECTORS_NUM_WORDS = 500; #500 words for testing.. (speeds stuff up a bit)
#WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)

#fake data
PICKLED_WORD_VECTORS_FILE = "data/fake_word_vectors.pkl.gz";
WORD_VECTORS_NUM_WORDS = 50; 
WORD_VECTORS_NUM_USERS = 20;


class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of 
    the dA at layer `i+1`. The first layer dA gets as input the input of 
    the SdA, and the hidden layer of the last dA represents the output. 
    Note that after pretraining, the SdA is dealt with as a normal MLP, 
    the dAs are only used to initialize the weights.
    """

    def __init__(self, numpy_rng, theano_rng = None, n_ins = WORD_VECTORS_NUM_WORDS, 
                 hidden_layers_sizes = [300,300], n_outs = 10, 
                 corruption_levels = [0.1, 0.1]):
        """ This class is made to support a variable number of layers. 

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial 
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is 
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain 
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        
        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each 
                                  layer
        """
        
        self.sigmoid_layers = []
        self.dA_layers      = []
        self.params         = []
        self.n_layers       = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))
        # allocate symbolic variables for the data
        self.x  = T.matrix('x')  # the data is presented as float valued vectors

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders 
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a 
        # denoising autoencoder that shares weights with that layer 
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing 
        # stochastich gradient descent on the MLP

        for i in xrange( self.n_layers ):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of 
            # the layer below or the input size if we are on the first layer
            if i == 0 :
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0 : 
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng   = numpy_rng, 
                                           input = layer_input, 
                                           n_in  = input_size, 
                                           n_out = hidden_layers_sizes[i],
                                           activation = T.nnet.sigmoid)
            # add the layer to our list of layers 
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the 
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)
        
            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng = numpy_rng, theano_rng = theano_rng, input = layer_input, 
                          n_visible = input_size, 
                          n_hidden  = hidden_layers_sizes[i],  
                          W = sigmoid_layer.W, bhid = sigmoid_layer.b)
            self.dA_layers.append(dA_layer)        


        ## We now need to add a logistic layer on top of the MLP
        #self.logLayer = LogisticRegression(\
        #                input = self.sigmoid_layers[-1].output,\
        #                 n_in = hidden_layers_sizes[-1], n_out = n_outs)
        #
        #self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

#  Need a function that estimates cost as a comparison to the x that was input


#
#        # compute the cost for second phase of training, 
#        # defined as the negative log likelihood 
#        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
#        # compute the gradients with respect to the model parameters
#        # symbolic variable that points to the number of errors made on the
#        # minibatch given by self.x and self.y
#        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one 
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on 
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of 
                              the dA layers
        '''

        # index to a [mini]batch
        index            = T.lscalar('index')   # index to a minibatch
        corruption_level = T.scalar('corruption')    # amount of corruption to use
        learning_rate    = T.scalar('lr')    # learning rate to use
        # number of batches
        n_batches = train_set_x.value.shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin+batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost,updates = dA.get_cost_updates( corruption_level, learning_rate)
            # compile the theano function    
            fn = theano.function( inputs = [index, 
                              theano.Param(corruption_level, default = 0.2),
                              theano.Param(learning_rate, default = 0.1)], 
                    outputs = cost, 
                    updates = updates,
                    givens  = {self.x :train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def predict(self, x, index, batch_size):

        # we use a matrix because we expect a minibatch of several examples,
        # each example being a row
        input_x = T.dmatrix(name = 'input') 
        activation_1 = T.dmatrix(name='activation_1')
        activation_2 = T.dmatrix(name='activation_2')
        
        dALayer = self.dA_layers[0];
        y       = dALayer.get_hidden_values(input_x)
        z       = dALayer.get_reconstructed_input(y)
        activation_1 = z; #next layer activation = input of last layer

        dALayer = self.dA_layers[1];
        y       = dALayer.get_hidden_values(activation_1)
        z       = dALayer.get_reconstructed_input(y)
        activation_2 = z; #next layer activation = input of last layer

        # compile the theano function    
        fn = theano.function( inputs = [input_x], 
                 outputs =  activation_2) 


        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin+batch_size
        input_x = x[batch_begin:batch_end];
        return fn(input_x);
            
    #def build_finetune_functions(self, datasets, batch_size, learning_rate):
    #    '''Generates a function `train` that implements one step of 
    #    finetuning, a function `validate` that computes the error on 
    #    a batch from the validation set, and a function `test` that 
    #    computes the error on a batch from the testing set
    #
    #    :type datasets: list of pairs of theano.tensor.TensorType
    #    :param datasets: It is a list that contain all the datasets;  
    #                     the has to contain three pairs, `train`, 
    #                     `valid`, `test` in this order, where each pair
    #                     is formed of two Theano variables, one for the 
    #                     datapoints, the other for the labels
    #
    #    :type batch_size: int
    #    :param batch_size: size of a minibatch
    #
    #    :type learning_rate: float
    #    :param learning_rate: learning rate used during finetune stage
    #    '''
    #
    #    (train_set_x) = datasets[0]
    #    (valid_set_x) = datasets[1]
    #    (test_set_x) = datasets[2]
    #
    #    # compute number of minibatches for training, validation and testing
    #    n_valid_batches = valid_set_x.value.shape[0] / batch_size
    #    n_test_batches  = test_set_x.value.shape[0]  / batch_size
    #
    #    index   = T.lscalar('index')    # index to a [mini]batch 
    #
    #    # compute the gradients with respect to the model parameters
    #    gparams = T.grad(self.finetune_cost, self.params)
    #
    #    # compute list of fine-tuning updates
    #    updates = {}
    #    for param, gparam in zip(self.params, gparams):
    #        updates[param] = param - gparam*learning_rate
    #
    #    train_fn = theano.function(inputs = [index], 
    #          outputs =   self.finetune_cost, 
    #          updates = updates,
    #          givens  = {
    #            self.x : train_set_x[index*batch_size:(index+1)*batch_size],
    #            self.y : train_set_y[index*batch_size:(index+1)*batch_size]})
    #
    #    test_score_i = theano.function([index], self.errors,
    #             givens = {
    #               self.x: test_set_x[index*batch_size:(index+1)*batch_size],
    #               self.y: test_set_y[index*batch_size:(index+1)*batch_size]})
    #
    #    valid_score_i = theano.function([index], self.errors,
    #          givens = {
    #             self.x: valid_set_x[index*batch_size:(index+1)*batch_size],
    #             self.y: valid_set_y[index*batch_size:(index+1)*batch_size]})
    #
    #    # Create a function that scans the entire validation set
    #    def valid_score():
    #        return [valid_score_i(i) for i in xrange(n_valid_batches)]
    #
    #    # Create a function that scans the entire test set
    #    def test_score():
    #        return [test_score_i(i) for i in xrange(n_test_batches)]
    #
    #    return train_fn, valid_score, test_score

    #added MKT
    def exportModel():
        return self.params;
    
    def loadModel(inpt_params):
        self.params=inpt_params;


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    # Load the dataset 
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_x):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        return shared_x
    
    test_set_x  = shared_dataset(test_set)
    valid_set_x = shared_dataset(valid_set)
    train_set_x = shared_dataset(train_set)

    rval = [train_set_x, valid_set_x, test_set_x]
    return rval

def save_model(sda):
    save_file=open(WEIGHTS_FILE,'wb')
    cPickle.dump(sda.params, save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(sda):
    save_file=open(WEIGHTS_FILE)
    sda.params = cPickle.load(save_file);
    save_file.close();
    return sda;
        

def train_SdA( finetune_lr = 0.1, pretraining_epochs = 1, \
              pretrain_lr = 0.001, training_epochs = 1000, \
              dataset=PICKLED_WORD_VECTORS_FILE, batch_size = 10):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage 
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer 

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data(dataset)

    train_set_x = datasets[0]
    valid_set_x = datasets[1]
    test_set_x  = datasets[2]


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class - go direct to 2 dimensions - hopefully real valued although to be honest who the heck knows at this point
    
    sda = SdA( numpy_rng = numpy_rng, n_ins =  train_set_x.value.shape[1], 
                      hidden_layers_sizes = [10,10],
                      n_outs =2)
    

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions( 
                                        train_set_x   = train_set_x, 
                                        batch_size    = batch_size ) 

    print '... pre-training the model'
    start_time = time.clock()  
    ## Pre-train layer-wise 
    corruption_levels = [.1,.2,.3]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append( pretraining_fns[i](index = batch_index, 
                         corruption = corruption_levels[i], 
                         lr = pretrain_lr ) )
            print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),numpy.mean(c)
 
    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))
    
    
    print 'saving model params'
    save_model(sda);
    
    print 'loading model params'
    load_model(sda);
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    #print '... getting the finetuning functions'
    #train_fn, validate_model, test_model = sda.build_finetune_functions ( 
    #            datasets = datasets, batch_size = batch_size, 
    #            learning_rate = finetune_lr) 
    #
    #print '... finetuning the model'
    ## early-stopping parameters
    #patience              = 10*n_train_batches # look as this many examples regardless
    #patience_increase     = 2.    # wait this much longer when a new best is 
    #                              # found
    #improvement_threshold = 0.995 # a relative improvement of this much is 
    #                              # considered significant
    #validation_frequency  = min(n_train_batches, patience/2)
    #                              # go through this many 
    #                              # minibatche before checking the network 
    #                              # on the validation set; in this case we 
    #                              # check every epoch 
    #
    #
    #best_params          = None
    #best_validation_loss = float('inf')
    #test_score           = 0.
    #start_time = time.clock()
    #
    #done_looping = False
    #epoch = 0
    #
    #while (epoch < training_epochs) and (not done_looping):
    #    for minibatch_index in xrange(n_train_batches):
    #        minibatch_avg_cost = train_fn(minibatch_index)
    #        iter    = epoch * n_train_batches + minibatch_index
    #
    #        if (iter+1) % validation_frequency == 0:
    #            validation_losses = validate_model()
    #            this_validation_loss = numpy.mean(validation_losses)
    #            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
    #               (epoch, minibatch_index+1, n_train_batches, \
    #                this_validation_loss*100.))
    #
    #
    #            # if we got the best validation score until now
    #            if this_validation_loss < best_validation_loss:
    #
    #                #improve patience if loss improvement is good enough
    #                if this_validation_loss < best_validation_loss *  \
    #                                            improvement_threshold :
    #                    patience = max(patience, iter * patience_increase)
    #
    #                # save best validation score and iteration number
    #                best_validation_loss = this_validation_loss
    #                best_iter = iter
    #
    #                # test it on the test set
    #                test_losses = test_model()
    #                test_score = numpy.mean(test_losses)
    #                print(('     epoch %i, minibatch %i/%i, test error of best '
    #                      'model %f %%') % 
    #                         (epoch, minibatch_index+1, n_train_batches,
    #                          test_score*100.))
    #
    #
    #        if patience <= iter :
    #            done_looping = True
    #            break
    #    epoch = epoch + 1
    #
    #end_time = time.clock()
    #print(('Optimization complete with best validation score of %f %%,'
    #       'with test performance %f %%') %  
    #             (best_validation_loss * 100., test_score*100.))
    #print >> sys.stderr, ('The training code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))


def load_and_run_SdA(batch_size = 10):


    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    
    print '..loading model params'
    sda = SdA( numpy_rng = numpy_rng)
    load_model(sda);
    
    # load some data to play with
    datasets = load_data(PICKLED_WORD_VECTORS_FILE)
    #train_set_x = datasets[0]
    #valid_set_x = datasets[1]
    test_set_x  = datasets[2]
    
    print '... running the model on test data'
    start_time = time.clock()  
    
    # compute number of minibatches for training, validation and testing
    n_batches = test_set_x.value.shape[0] / batch_size

    for index in xrange(n_batches):       
        out = sda.predict(test_set_x, index, batch_size);
        print "output:" + out;
        
    end_time = time.clock()
        
    
if __name__ == '__main__':
    train_SdA();
    #load_and_run_SdA();


