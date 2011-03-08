""
import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from hashmapd import *

import matplotlib.pyplot as plt

def load_data(dataset_file, n_ins):
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

    train_set_x = theano.shared(numpy.asarray(train_set[0], dtype=theano.config.floatX)) #shared_dataset(train_set)
    valid_set_x = theano.shared(numpy.asarray(valid_set[0], dtype=theano.config.floatX)) #shared_dataset(valid_set)
    test_set_x  = theano.shared(numpy.asarray(test_set[0], dtype=theano.config.floatX)) #shared_dataset(test_set)

    train_set_x_sums = theano.shared(numpy.asarray(numpy.array([train_set[2]]*n_ins).transpose(), dtype='int32'))
    valid_set_x_sums = theano.shared(numpy.asarray(numpy.array([train_set[2]]*n_ins).transpose(), dtype='int32'))
    test_set_x_sums  = theano.shared(numpy.asarray(numpy.array([train_set[2]]*n_ins).transpose(), dtype='int32'))

    rval = [train_set_x, valid_set_x, test_set_x, train_set_x_sums, valid_set_x_sums, test_set_x_sums]
    return rval
    
def train_SMH( finetune_lr = 0.3, pretraining_epochs = 100, \
              pretrain_lr = 0.01, method = 'cd', k = 1, noise_std_dev = 0, cost_method = 'squared_diff', training_epochs = 100, \
              dataset='data/truncated_mnist.pkl.gz', batch_size = 10, mid_layer_sizes=[200], inner_code_length=10, n_ins=784,
              skip_trace_images=False):

    datasets = load_data(dataset,n_ins)

    train_set_x = datasets[0]
    valid_set_x = datasets[1]
    test_set_x  = datasets[2]
    
    train_set_x_sums = datasets[3]
    valid_set_x_sums = datasets[4]
    test_set_x_sums  = datasets[5]

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
            train_set_x      = train_set_x,
            train_set_x_sums = train_set_x_sums,  
            batch_size       = batch_size,
            method           = method,
            k                = k)

    print '... pre-training the model'
    start_time = time.clock()
    
    plt.figure(1)
    
    ## Pre-train layer-wise 
    for i in xrange(smh.n_rbm_layers):
        
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    batch_cost = pretraining_fns[i](index = batch_index, lr = pretrain_lr )
                    c.append(batch_cost)
                
                print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),numpy.mean(c)

#                debugging
#
#                if i == 0:
#                    # plot weights
#                    plt.subplot(311)
#                    m = smh.rbm_layers[0].W.value.argmax()/smh.rbm_layers[0].W.value.shape[1]
#                    plt.plot(smh.rbm_layers[0].W.value[0],color='red',linestyle='None',marker='o')
#                    plt.plot(smh.rbm_layers[0].W.value[1],color='blue',linestyle='None',marker='o')
#                    plt.plot(smh.rbm_layers[0].W.value[2],color='green',linestyle='None',marker='o')
#                    plt.plot(smh.rbm_layers[0].W.value[m],color='yellow',linestyle='None',marker='D')
#                    print numpy.max(numpy.abs(smh.rbm_layers[0].W.value))
#                    plt.ylabel('weights')
#                    plt.xlabel('units')
#                    # plot vis biases
#                    plt.subplot(312)
#                    plt.plot(smh.rbm_layers[0].vbias.value,'bo')
#                    plt.ylabel('vis biases')
#                    plt.xlabel('units')
#                    # plot hidd biases
#                    plt.subplot(313)
#                    plt.plot(smh.rbm_layers[0].hbias.value,'go')
#                    plt.ylabel('hidd biases')
#                    plt.xlabel('units')
#                    # print largest data * weight value (if closing in on 30, then will have issues)
#                    m = train_set_x.value.argmax()/train_set_x.value.shape[1];
#                    print numpy.max(numpy.abs(numpy.dot(train_set_x.value[m],smh.rbm_layers[0].W.value)+smh.rbm_layers[0].hbias.value))
#                    # show plot
#                    plt.show()
    
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))
    
    
    smh.unroll_layers(cost_method,noise_std_dev);
    output_trace_info(smh, datasets,'b4_finetuning',skip_trace_images)
 
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

    output_trace_info(smh, datasets,'after_finetuning',skip_trace_images)
   
    return smh

def output_trace_info(smh, datasets, prefix, skip_trace_images):
    
    
    ########################
    # OUTPUT WEIGHTS       #
    ########################
    
    #output state of weights
    for layer in xrange(smh.n_sigmoid_layers):
        sigmoid_layer = smh.sigmoid_layers[layer]
        sigmoid_layer.export_weights_image('trace/%s_weights_%i.png'%(prefix,layer))
    
    if skip_trace_images:
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
    cPickle.dump(smh.export_model(), save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(cost_method, first_layer_type = 'bernoulli', n_ins=784,  mid_layer_sizes = [200],
                    inner_code_length = 10, weights_file=DEFAULT_WEIGHTS_FILE):
    numpy_rng = numpy.random.RandomState(212)
    smh = SMH(numpy_rng = numpy_rng,  first_layer_type = first_layer_type, mid_layer_sizes = mid_layer_sizes, inner_code_length = inner_code_length, n_ins = n_ins)
    smh.unroll_layers(cost_method,0); #need to unroll before loading model otherwise doesn't work
    save_file=open(weights_file, 'rb')
    smh_params = cPickle.load(save_file)
    save_file.close()
    smh.load_model(smh_params)
    return smh

def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])
    #validate_config(cfg)
    
    data_file = cfg.input.train_data
    n_ins = cfg.shape.input_vector_length
   
    weights_file = cfg.train.weights_file
    skip_trace_images = cfg.train.skip_trace_images
   
    input_vector_length = cfg.shape.input_vector_length
    mid_layer_sizes = list(cfg.shape.mid_layer_sizes)
    inner_code_length = cfg.shape.inner_code_length
    
    train_batch_size = cfg.train.train_batch_size
    pretraining_epochs = cfg.train.pretraining_epochs
    training_epochs = cfg.train.training_epochs
    
    method = cfg.train.method
    k = cfg.train.k
    first_layer_type = cfg.train.first_layer_type
    noise_std_dev = cfg.train.noise_std_dev
    cost_method = cfg.train.cost
    
    smh = train_SMH(dataset = data_file,
                    batch_size = train_batch_size, 
                    pretraining_epochs = pretraining_epochs,
                    training_epochs = training_epochs,
                    mid_layer_sizes = mid_layer_sizes,
                    inner_code_length = inner_code_length,
                    n_ins=input_vector_length,
                    method = method,
                    k = k,
                    noise_std_dev = noise_std_dev,
                    cost_method = cost_method,
                    skip_trace_images = skip_trace_images)
    
    save_model(smh, weights_file=weights_file)
    
    #double check that save/load worked OK
    datasets = load_data(data_file,input_vector_length)
    output_trace_info(smh, datasets[:3], 'test_weights_b4_restore', skip_trace_images)
    
    smh2 = load_model(cost_method = cost_method, first_layer_type = first_layer_type, n_ins=n_ins,  mid_layer_sizes = mid_layer_sizes,
                    inner_code_length = inner_code_length, weights_file=weights_file)
    
    output_trace_info(smh2, datasets[:3], 'test_weights_restore', skip_trace_images)
    
if __name__ == '__main__':
    sys.exit(main())
    

    
        
    
    
