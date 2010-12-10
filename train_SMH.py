""
import os, sys

import numpy, time, cPickle, gzip, PIL.Image

import theano
import theano.tensor as T
from SMH import SMH
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images

#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#truncatd from full set for easier test
#TRUNCATED_MNIST_FILE = "data/truncated_mnist.pkl.gz"
#as above, then normalized by row, so that testing rbm_softmax makes sense
#TRUNCATED_MNIST_FILE = "data/truncated_mnist.pkl.gz"
#TRUNCATED_NORMALIZED_MNIST_FILE = "data/truncated_normalized_mnist.pkl.gz"
UNSUPERVISED_MNIST = 'data/truncated_unsupervised_mnist.pkl.gz'
FAKE_IMG_DATA_FILE = "data/fake_img_data.pkl.gz"
UNSUPERVISED_MNIST_WEIGHTS_FILE = "data/unsuperivsed_mnist_weights.pkl.gz"
NUM_PIXELS = 784;


WORD_VECTORS_FILE = "data/word_vectors.pkl.gz";
WORD_VECTORS_NUM_WORDS = 3000; #number of different words in above file ('word_id' column is allowed to be *up to and including* this number)
WORD_VECTORS_NUM_USERS = 5285; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)
WORD_VECTORS_WEIGHTS_FILE = "data/word_vectors_weights.pkl.gz"

SKIP_TRACE_OUTPUT = False

 
def load_data(dataset_file):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data from ' + dataset_file

    # Load the dataset  - expecting both supervised and unsupervised data to be supplied (in pairs)
    f = gzip.open(dataset_file,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_set_x = theano.shared(numpy.asarray(train_set, dtype=theano.config.floatX)) #shared_dataset(train_set)
    valid_set_x = theano.shared(numpy.asarray(valid_set, dtype=theano.config.floatX)) #shared_dataset(valid_set)
    test_set_x  = theano.shared(numpy.asarray(test_set, dtype=theano.config.floatX)) #shared_dataset(test_set)

    rval = [train_set_x, valid_set_x, test_set_x]
    return rval


def save_model(smh):
    save_file=open(WEIGHTS_FILE,'wb')
    cPickle.dump(smh.params, save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(smh):
    save_file=open(WEIGHTS_FILE)
    smh.params = cPickle.load(save_file);
    save_file.close();
    return smh;
    
def train_SMH( finetune_lr = 0.3, pretraining_epochs = 100, \
              pretrain_lr = 0.01, k = 1, training_epochs = 100, \
              dataset='data/truncated_mnist.pkl.gz', batch_size = 10, mid_layer_sizes=[200], inner_code_length=10, n_ins=784):

    datasets = load_data(dataset)

    train_set_x = datasets[0]
    valid_set_x = datasets[1]
    test_set_x   = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    smh = SMH(numpy_rng = numpy_rng, inner_code_length=inner_code_length, mid_layer_sizes=mid_layer_sizes, n_ins = n_ins)      
    

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = smh.pretraining_functions(
            train_set_x   = train_set_x, 
            batch_size    = batch_size,
            k             = k) 

    print '... pre-training the model'
    start_time = time.clock()  
    ## Pre-train layer-wise 
    for i in xrange(smh.n_rbm_layers):
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

    output_trace_info(smh, datasets,'b4_finetuning')


    ########################
    # FINETUNING THE MODEL #
    ########################
    #
    ## get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = smh.build_finetune_functions ( 
                datasets = datasets, batch_size = batch_size, 
                learning_rate = finetune_lr) 
    
    print '... finetuning the model'
    # early-stopping parameters
    patience              = 4*n_train_batches # look as this many examples regardless
    patience_increase     = 2.    # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.9995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_batches, patience/2)
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 
    
    
    best_params          = None
    best_validation_loss = float('inf')
    test_score           = 0.
    start_time = time.clock()
    
    done_looping = False
    epoch = 0
    
    while (epoch < training_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in xrange(n_train_batches):
    
        minibatch_avg_cost = train_fn(minibatch_index)
        iter    = epoch * n_train_batches + minibatch_index
        
        if (iter+1) % validation_frequency == 0: 
            
            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                   (epoch, minibatch_index+1, n_train_batches, \
                    this_validation_loss*100.))
        
        
            #if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
        
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)
        
                #save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter
        
                #test it on the test set
                test_losses = test_model()
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f %%') % 
                             (epoch, minibatch_index+1, n_train_batches,
                              test_score*100.))
        
    
        if patience <= iter :
                done_looping = True
                break
    
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print >> sys.stderr, ('The fine tuning code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))

    output_trace_info(smh, datasets,'after_finetuning')
   
    return smh

def output_trace_info(smh, datasets, prefix):
    
    
    
    ########################
    # OUTPUT WEIGHTS       #
    ########################
    
    #output state of weights
    for layer in xrange(smh.n_unrolled_layers):
        sigmoid_layer = smh.sigmoid_layers[layer]
        sigmoid_layer.export_weights_image('trace/%s_weights_%i.png'%(prefix,layer))
    
    if SKIP_TRACE_OUTPUT:
        return
   
        
    #################################
    # RUN RECONSTRUCTION SAMPLES    #
    #################################
   
    train_set_x = datasets[0]
    valid_set_x = datasets[1]
    test_set_x   = datasets[2]
       
    data_x = test_set_x.value #[index*batch_size:(index+1)*batch_size];
    output_y = smh.output_given_x(data_x);
    
    output_y = smh.output_given_x(test_set_x.value);
    
    # Plot image and reconstrution 
    image = PIL.Image.fromarray(tile_raster_images( X = data_x,
             img_shape = (28,28),tile_shape = (10,10), 
             tile_spacing=(1,1)))
    image.save('trace/%s_input.png'%prefix)
    
    image = PIL.Image.fromarray(tile_raster_images( X = output_y,
             img_shape = (28,28),tile_shape = (10,10), 
             tile_spacing=(1,1)))
    image.save('trace/%s_reconstruction.png'%prefix)


DEFAULT_WEIGHTS_FILE='data/last_smh_model_params.pkl.gz'
def save_model(smh, weights_file=DEFAULT_WEIGHTS_FILE):
    save_file=open(weights_file,'wb')
    cPickle.dump(smh.exportModel(), save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(n_ins=784,  mid_layer_sizes = [200],
                    inner_code_length = 10, weights_file=DEFAULT_WEIGHTS_FILE):
    numpy_rng = numpy.random.RandomState(212)
    smh = SMH(numpy_rng = numpy_rng,  mid_layer_sizes = mid_layer_sizes, inner_code_length = inner_code_length, n_ins = n_ins)
    save_file=open(weights_file)
    smh_params = cPickle.load(save_file)
    save_file.close()
    smh.loadModel(smh_params)
    return smh
   
if __name__ == '__main__':
   
    #data_file = UNSUPERVISED_MNIST
    #weights_file = UNSUPERVISED_MNIST_WEIGHTS_FILE
    #n_ins = NUM_PIXELS
 
    data_file = WORD_VECTORS_FILE
    weights_file = WORD_VECTORS_WEIGHTS_FILE
    n_ins = WORD_VECTORS_NUM_WORDS
    SKIP_TRACE_OUTPUT = True
    
    smh = train_SMH(dataset=data_file,
                    batch_size=10, 
                    pretraining_epochs = 100,
                    training_epochs = 1000,
                    mid_layer_sizes = [1000,200],
                    inner_code_length = 30,
                    n_ins=n_ins)
    
    save_model(smh, weights_file=weights_file)
    
    #double check that save/load worked OK
    datasets = load_data(data_file)
    output_trace_info(smh, datasets, 'test_weights_b4_restore')
    
    smh2 = load_model(n_ins=n_ins,  mid_layer_sizes = [1000,200],
                    inner_code_length = 30, weights_file=weights_file)
    output_trace_info(smh2, datasets, 'test_weights_restore')
    
    
        
    
    