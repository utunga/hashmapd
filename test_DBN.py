""
import os

import numpy, time, cPickle, gzip, os, sys

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from mlp import HiddenLayer
#from rbm import RBM
#from logistic_sgd import LogisticRegression
from DBN import DBN

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

def save_model(dbn):
    save_file=open(WEIGHTS_FILE,'wb')
    cPickle.dump(dbn.params, save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(dbn):
    save_file=open(WEIGHTS_FILE)
    dbn.params = cPickle.load(save_file);
    save_file.close();
    return dbn;
        
def train_DBN( finetune_lr = 0.1, pretraining_epochs = 10, \
              pretrain_lr = 0.01, k = 1, training_epochs = 10, \
              dataset='../data/mnist_truncated.pkl.gz', batch_size = 10, n_ins=784):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage 
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer 
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """


    datasets = load_data(dataset)

    train_set_x = datasets[0]
    valid_set_x = datasets[1]
    test_set_x = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng = numpy_rng, n_ins = n_ins, 
              hidden_layers_sizes = [10,10,10],
              n_outs = 10)
    

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(
            train_set_x   = train_set_x, 
            batch_size    = batch_size,
            k             = k) 

    print '... pre-training the model'
    start_time = time.clock()  
    ## Pre-train layer-wise 
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index = batch_index, 
                         lr = pretrain_lr ) )
            print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),numpy.mean(c)
 
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))
    
    ########################
    # FINETUNING THE MODEL #
    ########################
    #
    ## get the training, validation and testing function for the model
    #print '... getting the finetuning functions'
    #train_fn, validate_model, test_model = dbn.build_finetune_functions ( 
    #            datasets = datasets, batch_size = batch_size, 
    #            learning_rate = finetune_lr) 
    #
    #print '... finetunning the model'
    ## early-stopping parameters
    #patience              = 4*n_train_batches # look as this many examples regardless
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
    #  epoch = epoch + 1
    #  for minibatch_index in xrange(n_train_batches):
    #
    #    minibatch_avg_cost = train_fn(minibatch_index)
    #    iter    = epoch * n_train_batches + minibatch_index
    #
    #    if (iter+1) % validation_frequency == 0: 
    #        
    #        validation_losses = validate_model()
    #        this_validation_loss = numpy.mean(validation_losses)
    #        print('epoch %i, minibatch %i/%i, validation error %f %%' % \
    #               (epoch, minibatch_index+1, n_train_batches, \
    #                this_validation_loss*100.))
    #
    #
    #        # if we got the best validation score until now
    #        if this_validation_loss < best_validation_loss:
    #
    #            #improve patience if loss improvement is good enough
    #            if this_validation_loss < best_validation_loss *  \
    #                   improvement_threshold :
    #                patience = max(patience, iter * patience_increase)
    #
    #            # save best validation score and iteration number
    #            best_validation_loss = this_validation_loss
    #            best_iter = iter
    #
    #            # test it on the test set
    #            test_losses = test_model()
    #            test_score = numpy.mean(test_losses)
    #            print(('     epoch %i, minibatch %i/%i, test error of best '
    #                  'model %f %%') % 
    #                         (epoch, minibatch_index+1, n_train_batches,
    #                          test_score*100.))
    #
    #
    #    if patience <= iter :
    #            done_looping = True
    #            break
    #
    #end_time = time.clock()
    #print(('Optimization complete with best validation score of %f %%,'
    #       'with test performance %f %%') %  
    #             (best_validation_loss * 100., test_score*100.))
    #print >> sys.stderr, ('The fine tuning code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))

    save_model(dbn);
    return dbn


if __name__ == '__main__':
    dbn = train_DBN(dataset=PICKLED_WORD_VECTORS_FILE, batch_size=10, n_ins=WORD_VECTORS_NUM_WORDS)


#WORD_VECTORS_NUM_USERS = 20;