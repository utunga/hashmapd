import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import theano

import matplotlib.pyplot as plt

def get_git_home():
    testpath = '.'
    while not '.git' in os.listdir(testpath) and not os.path.abspath(testpath) == '/':
        testpath = os.path.sep.join(('..', testpath))
    if not os.path.abspath(testpath) == '/':
        return os.path.abspath(testpath)
    else:
        raise ValueError, "Not in git repository"

HOME = get_git_home()
sys.path.append(HOME)

from hashmapd.load_config import LoadConfig, DefaultConfig
from hashmapd.SMH import SMH

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
    shared_x = numpy.asarray(x, dtype=theano.config.floatX)
    # build a replcated 2d array of sums so operations can be performed efficiently
    shared_x_sums = numpy.asarray(numpy.array([x_sums]*(x.shape[1])).transpose(), dtype=theano.config.floatX)
    
    rval = [shared_x,shared_x_sums]
    return rval
    

def train_SMH(finetune_lr = 0.3, pretraining_epochs = 100, pretrain_lr = 0.01, training_epochs = 100, \
              first_layer_type = 'bernoulli', method = 'cd', k = 1, noise_std_dev = 0, cost_method = 'squared_diff', \
              dataset_info='data', \
              batch_size = 10, mid_layer_sizes=[200], inner_code_length=10, n_ins=784, \
              skip_trace_images=False, weights_file=None):
    
    # load info file to determine how files are stored
    dataset_postfix = '.pkl.gz';
   
    info = LoadConfig(dataset_info)['info']

    # if only one training file, load all the data at once
    if info['n_training_files'] == 1:
        training_data = load_data(info['training_prefix']+'_0'+dataset_postfix)
        validation_data = load_data(info['validation_prefix']+'_0'+dataset_postfix)
        testing_data = load_data(info['testing_prefix']+'_0'+dataset_postfix)
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    smh = SMH(numpy_rng=numpy_rng, first_layer_type=first_layer_type, mean_doc_size=info['mean_doc_size'], \
              inner_code_length=inner_code_length, mid_layer_sizes=mid_layer_sizes, n_ins = n_ins) 
    
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = smh.pretraining_functions(
            batch_size       = batch_size,
            method           = method,
            k                = k)

    print '... pre-training the model'
    start_time = time.clock()
    
    ## Pre-train layer-wise 
    for i in xrange(smh.n_rbm_layers):
        
        # go through pretraining epochs 
        for epoch in xrange(pretraining_epochs):
            
                c = []
                # go through the training set
                for file_index in xrange(info['n_training_files']):
                    if info['n_training_files'] > 1: training_data = load_data(info['training_prefix']+'_'+str(file_index)+dataset_postfix)
                    no_batches = info['batches_per_file']
                    # determine the number of batches in file
                    if (file_index+1)*info['batches_per_file'] > info['n_training_batches']:
                        no_batches = max(0,info['n_training_batches']-(file_index*info['batches_per_file']))
                    # perform pretraining on each batch
                    for batch_index in xrange(no_batches):
                        batch_cost = pretraining_fns[i](train_set_x = training_data[0][batch_index*batch_size:(batch_index+1)*(batch_size),:], \
                            train_set_x_sums = training_data[1][batch_index*batch_size:(batch_index+1)*(batch_size),:], lr = pretrain_lr )
                        c.append(batch_cost)
                    
                    if info['n_training_files'] > 1: training_data = None
                
                print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),numpy.mean(c)

#===============================================================================
#               Matplotlib debugging:
#
#               if i == 0:
#                   # plot weights (first three vis units, as well as the largest/smallest) 
#                   plt.subplot(311)
#                   max_i = smh.rbm_layers[0].W.value.argmax()/smh.rbm_layers[0].W.value.shape[1]
#                   plt.plot(smh.rbm_layers[0].W.value[0],color='red',linestyle='None',marker='o')
#                   plt.plot(smh.rbm_layers[0].W.value[1],color='blue',linestyle='None',marker='o')
#                   plt.plot(smh.rbm_layers[0].W.value[2],color='green',linestyle='None',marker='o')
#                   plt.plot(smh.rbm_layers[0].W.value[max_i],color='yellow',linestyle='None',marker='D')
#                   print numpy.max(numpy.abs(smh.rbm_layers[0].W.value)) # print the max weight value
#                   plt.ylabel('weights')
#                   plt.xlabel('units')
#
#                   # plot vis biases
#                   plt.subplot(312)
#                   plt.plot(smh.rbm_layers[0].vbias.value,'bo')
#                   plt.ylabel('vis biases')
#                   plt.xlabel('units')
#
#                   # plot hidd biases
#                   plt.subplot(313)
#                   plt.plot(smh.rbm_layers[0].hbias.value,'go')
#                   plt.ylabel('hidd biases')
#                   plt.xlabel('units')
#
#                   # print largest data * weight value (if 20+, may have issues?)
#                   m = train_set_x.value.argmax()/train_set_x.value.shape[1];
#                   print numpy.max(numpy.abs(numpy.dot(train_set_x.value[m],smh.rbm_layers[0].W.value)+smh.rbm_layers[0].hbias.value))
#
#                   # show plot
#                   plt.show()
#===============================================================================
    
    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))
    
    smh.unroll_layers(cost_method,noise_std_dev);
    
    # save model after pretraining
    if weights_file is not None:
        save_model(smh, weights_file=weights_file)
    
    # load in first file of test data for output trace
    if info['n_training_files'] > 1: testing_data = load_data(info['testing_prefix']+'_'+'0'+dataset_postfix)
    output_trace_info(smh, testing_data[0],'b4_finetuning',skip_trace_images)
    if info['n_training_files'] > 1: testing_data = None # ensure test data is unloaded
    
    ########################
    # FINETUNING THE MODEL #
    ########################
    
    ## get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model_i, test_model_i = smh.build_finetune_functions ( 
                batch_size = batch_size, 
                learning_rate = finetune_lr) 
    
    print '... finetuning the model'
    # early-stopping parameters
    patience              = 4*info['n_training_batches'] # look as this many examples regardless
    patience_increase     = 2.    # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.9995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(info['n_training_batches'], patience/2)
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
        minibatch_iter = 0
        
        # go through the training set
        for file_index in xrange(info['n_training_files']):
            if info['n_training_files'] > 1: training_data = load_data(info['training_prefix']+str(file_index)+dataset_postfix)
            no_batches = info['batches_per_file']
            # determine the number of batches in file
            if (file_index+1)*info['batches_per_file'] > info['n_training_batches']:
                no_batches = max(0,info['n_training_batches']-(file_index*info['batches_per_file']))
            
            # perform finetuning on each batch
            for batch_index in xrange(no_batches):
                minibatch_avg_cost = train_fn(train_set_x = training_data[0][batch_index*batch_size:(batch_index+1)*(batch_size),:], \
                                        train_set_x_sums = training_data[1][batch_index*batch_size:(batch_index+1)*(batch_size),:])
                
                iter    = epoch * info['n_training_batches'] + minibatch_iter
                minibatch_iter += 1
                
                # check if it is time to perform validation
                if (iter+1) % validation_frequency == 0:
                    if info['n_training_files'] > 1: training_data = None # ensure training data is unloaded from memory
                    
                    # go through the validation set
                    validation_losses = numpy.array([]);
                    for v_file_index in xrange(info['n_validation_files']):
                        if info['n_training_files'] > 1: validation_data = load_data(info['validation_prefix']+str(v_file_index)+dataset_postfix)
                        no_v_batches = info['batches_per_file']
                        # determine the number of batches in file
                        if (v_file_index+1)*info['batches_per_file'] > info['n_validation_batches']:
                            no_v_batches = max(0,info['n_validation_batches']-(v_file_index*info['batches_per_file']))
                        # determine the validation loss on this batch
                        for v_batch_index in xrange(no_v_batches):
                            validation_loss_i = validate_model_i(valid_set_x = validation_data[0][v_batch_index*batch_size:(v_batch_index+1)*(batch_size),:], \
                                                    valid_set_x_sums = validation_data[1][v_batch_index*batch_size:(v_batch_index+1)*(batch_size),:])
                            validation_losses = numpy.append(validation_losses,[validation_loss_i])
                    
                    if info['n_training_files'] > 1: validation_data = None # ensure validation data is unloaded from memory
                    
                    # determine total validation score
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                       (epoch, minibatch_iter, info['n_training_batches'], \
                        this_validation_loss*100.))
                    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold :
                            patience = max(patience, iter * patience_increase)
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        
                        # go through the test set
                        test_losses = numpy.array([]);
                        for t_file_index in xrange(info['n_testing_files']):
                            if info['n_training_files'] > 1: testing_data = load_data(info['testing_prefix']+str(v_file_index)+dataset_postfix)
                            no_t_batches = info['batches_per_file']
                            # determine the number of batches in file
                            if (t_file_index+1)*info['batches_per_file'] > info['n_testing_batches']:
                                no_t_batches = max(0,info['n_testing_batches']-(t_file_index*info['batches_per_file']))
                            # determine the test loss on this batch
                            for t_batch_index in xrange(no_t_batches):
                                test_loss_i = test_model_i(test_set_x = testing_data[0][t_batch_index*batch_size:(t_batch_index+1)*(batch_size),:], \
                                                        test_set_x_sums = testing_data[1][t_batch_index*batch_size:(t_batch_index+1)*(batch_size),:])
                                test_losses = numpy.append(test_losses,[test_loss_i])
                        
                        if info['n_training_files'] > 1: testing_data = None # ensure test data is unloaded from memory
                        
                        # determine total test score
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of best '
                              'model %f %%') % 
                                   (epoch, minibatch_iter, info['n_training_batches'], \
                                    test_score*100.))
                    
                    # reload the training data, and continue fine tuning
                    if info['n_training_files'] > 1: training_data = load_data(info['training_prefix']+str(file_index)+dataset_postfix)
                
                # check if we are done
                if patience <= iter :
                    done_looping = True
                    break
    
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print >> sys.stderr, ('The fine tuning code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((end_time-start_time)/60.))

    if info['n_training_files'] > 1: testing_data = load_data(info['testing_prefix']+'0'+dataset_postfix)
    output_trace_info(smh, testing_data[0],'after_finetuning',skip_trace_images)
    if info['n_training_files'] > 1: testing_data = None # ensure test data is unloaded
    
    return smh

def output_trace_info(smh, testing_data_x, prefix, skip_trace_images):
    
    
    ########################
    # OUTPUT WEIGHTS       #
    ########################
    
    #output state of weights
    for layer in xrange(smh.n_sigmoid_layers):
        sigmoid_layer = smh.sigmoid_layers[layer]
        try:
            os.makedirs('trace')
        except OSError:
            pass
        try:
            sigmoid_layer.export_weights_image('trace/%s_weights_%i.png'%(prefix,layer))
        except IOError:
            pass
    
    if skip_trace_images:
        return
   
        
    #################################
    # RUN RECONSTRUCTION SAMPLES    #
    #################################
    
    data_x = testing_data_x #[index*batch_size:(index+1)*batch_size];
    output_y = smh.output_given_x(data_x); # output_y = smh.output_given_x(test_set_x.value);
    
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

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)

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
    
    smh = train_SMH(batch_size = train_batch_size, 
                    pretraining_epochs = pretraining_epochs,
                    training_epochs = training_epochs,
                    mid_layer_sizes = mid_layer_sizes,
                    inner_code_length = inner_code_length,
                    n_ins=input_vector_length,
                    first_layer_type = first_layer_type,
                    method = method,
                    k = k,
                    noise_std_dev = noise_std_dev,
                    cost_method = cost_method,
                    skip_trace_images = skip_trace_images,
                    weights_file=weights_file)
    
    #double check that save/load worked OK
    info = LoadConfig('data')['info']
    testing_data = load_data(info['testing_prefix']+'_0.pkl.gz')
    
    save_model(smh, weights_file=weights_file)
    output_trace_info(smh, testing_data[0][:3], 'test_weights_b4_restore', skip_trace_images)
    
    smh2 = load_model(cost_method = cost_method, first_layer_type = first_layer_type, n_ins=n_ins,  mid_layer_sizes = mid_layer_sizes,
                    inner_code_length = inner_code_length, weights_file=weights_file)
    output_trace_info(smh2, testing_data[0][:3], 'test_weights_restore', skip_trace_images)
    
    

    
        
    
    
