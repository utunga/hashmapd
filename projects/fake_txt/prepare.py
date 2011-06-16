import sys
import getopt
import os
import csv
import cPickle
import gzip
import theano
import time, PIL.Image
import numpy as np

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


from hashmapd.load_config import LoadConfig, DefaultConfig, dict_to_cfg

TRAINING_FILE = "training_data"
VALIDATION_FILE = "validation_data"
TESTING_FILE = "testing_data"
PICKLED_TYPE = ".pkl.gz"
#
#def read_user_word_pixels(cfg):
#    """
#    Reads in data from user_word_vectors.csv
#    """
#    print "attempting to read " + cfg.input.csv_data
#    
#    raw_pixels = np.zeros((cfg.input.number_of_examples, cfg.shape.input_vector_length), dtype=theano.config.floatX) #store as float so that normalized_counts uses float math
#    
#    vectorReader = csv.DictReader(open(cfg.input.csv_data, 'rb'), delimiter=',')
#    iter=0
#    for row in vectorReader:
#        if iter % 100==0:
#            print 'reading row '+ str(iter) + '..' #MKT: presumably a nicer way to do this
#        iter += 1
#        user_id = int(row['user_id'])
#        word_id = int(row['word_id'])
#        pixel = float(row['pixel'])
#        raw_pixels[user_id,word_id] = pixel
#    
#    #total_user_pixels = raw_pixels.sum(axis=1)
#    #normalized_pixels = (raw_pixels.transpose()/total_user_pixels).transpose()
#    
#    print 'done reading input'
#    print raw_pixels
#    return raw_pixels
#    
def read_user_word_counts(cfg):
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + cfg.raw.csv_data
    
    raw_counts = np.zeros((cfg.input.number_of_examples, cfg.shape.input_vector_length), dtype=theano.config.floatX) #store as float so that normalized_counts uses float math
    vectorReader = csv.DictReader(open(cfg.raw.csv_data, 'rb'), delimiter=',')
    iter=0
    for row in vectorReader:
        if iter % 10000==0:  #MKT: presumably a nicer way to do this ?
            print 'reading row '+ str(iter) + '..'
        iter += 1
        user_id = int(row['user_id'])
        word_id = int(row['word_id'])
        count = int(row['count'])
        assert user_id >= 0 and word_id >= 0 and count >= 0
        raw_counts[user_id,word_id] = count
    
    print 'done reading input'
    return raw_counts

def read_user_labels(cfg):
    """
    Reads in data from fake_txt_users.csv
    """
    print "attempting to read " + cfg.raw.users_file
    
    users = []
    vectorReader = csv.DictReader(open(cfg.raw.users_file, 'rb'), delimiter=',')
    iter=0
    for row in vectorReader:
        if iter % 10000==0:  #MKT: presumably a nicer way to do this ?
            print 'reading row '+ str(iter) + '..'
        iter += 1
        user_id = int(row['user_id'])
        user_name = row['screen']
        users.append((user_id, user_name))
    
    print 'done reading input'
    return dict(users)

def validate_config(cfg):
    
    num_examples = cfg.input.number_of_examples
    train_cutoff = cfg.input.number_for_training
    validate_cutoff = train_cutoff+cfg.input.number_for_validation
    test_cutoff = validate_cutoff+cfg.input.number_for_testing
    
    assert test_cutoff<=num_examples, \
        "total of number of examples should be more than cases for train, validate and test"
    assert 0<train_cutoff, \
        "config fail, number_for_training needs to be > 0"
    assert train_cutoff<=validate_cutoff, \
        "config fail, number_for_validation should be > 0"
    assert validate_cutoff<=test_cutoff, \
        "config fail, number_for_testing should be >= 0"
    
def get_filename(name, number, filetype=PICKLED_TYPE, extsep='.'):
    return os.path.join("data", "%s_%s%s"%(name, number, filetype))

def normalize_and_output_pickled_data(cfg, raw_counts, user_labels):
   
    print "outputting full data set"
    
    num_examples = cfg.input.number_of_examples
    train_cutoff = cfg.input.number_for_training
    validate_cutoff = train_cutoff+cfg.input.number_for_validation
    test_cutoff = validate_cutoff+cfg.input.number_for_testing
    batch_size = cfg.train.train_batch_size
    
    print raw_counts
    train_set_x = raw_counts[0:train_cutoff]
    train_sums = train_set_x.sum(axis=1)
    mean_doc_size = train_sums.mean()
    #print train_set_x
    #print train_sums
    
    if (cfg.input.number_for_validation ==0):
        print 'WARNING: no examples set aside for validation, copying train set data for validation (as a quick hack only)'
        valid_set_x = train_set_x
    else:
        valid_set_x = raw_counts[train_cutoff:validate_cutoff]
    valid_sums = valid_set_x.sum(axis=1)

    if (cfg.input.number_for_testing ==0):
        print 'WARNING: no examples set aside for testing, copying validation set data for training (as a quick hack only)'
        test_set_x = valid_set_x
    else:
        test_set_x = raw_counts[validate_cutoff:test_cutoff]
    test_sums = test_set_x.sum(axis=1)
   
    print '...  pickling and zipping train/validate/test data to data directory'
    
    train_file = gzip.open(get_filename(TRAINING_FILE, 0),'wb')
    valid_file = gzip.open(get_filename(VALIDATION_FILE, 0),'wb')
    test_file = gzip.open(get_filename(TESTING_FILE, 0), 'wb')
    
    if (cfg.train.first_layer_type=='poisson'):
        cPickle.dump((train_set_x,train_sums,np.zeros(train_sums.shape,dtype=theano.config.floatX)), train_file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump((valid_set_x,valid_sums,np.zeros(valid_sums.shape,dtype=theano.config.floatX)), valid_file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump((test_set_x,test_sums,np.zeros(test_sums.shape,dtype=theano.config.floatX)), test_file, cPickle.HIGHEST_PROTOCOL)
    else:
        cPickle.dump((normalize_data_x(train_set_x,train_sums,'training'),train_sums,np.zeros(train_sums.shape,dtype=theano.config.floatX)), train_file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump((normalize_data_x(valid_set_x,valid_sums,'validation'),valid_sums,np.zeros(valid_sums.shape,dtype=theano.config.floatX)), valid_file, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump((normalize_data_x(test_set_x,test_sums,'testing'),test_sums,np.zeros(test_sums.shape,dtype=theano.config.floatX)), test_file, cPickle.HIGHEST_PROTOCOL)
    
    train_file.close()
    valid_file.close()
    test_file.close() 
        
    data_info = {'training_prefix': os.path.join('data', TRAINING_FILE),
        'n_training_files': 1,
        'n_training_batches':train_cutoff/batch_size,
        'validation_prefix':  os.path.join('data', VALIDATION_FILE),
        'n_validation_files': 1,
        'n_validation_batches': validate_cutoff/batch_size,
        'testing_prefix':  os.path.join('data', TESTING_FILE),
        'n_testing_files': 1,
        'n_testing_batches': test_cutoff/batch_size,
        'batches_per_file': (train_cutoff+validate_cutoff+test_cutoff)/batch_size,
        'mean_doc_size': mean_doc_size,
    }

    dict_to_cfg(data_info, 'info', 'data.cfg')

    print '...  pickling and zipping render_data to '+ cfg.input.render_data
    render_data = normalize_data_x(train_set_x,train_sums,'training')[0:num_examples]
    f = gzip.open(cfg.input.render_data,'wb')
    
    render_file_has_labels = cfg.input.render_data_has_labels
    data = render_data
    if (render_file_has_labels):
        labels = user_labels.items()
        labels.sort()
        labels = labels[0:num_examples] #FIXME makes dangerous assumption that user_labels has same ordering as render_data
        data = (render_data, labels)
        
    cPickle.dump(data,f, cPickle.HIGHEST_PROTOCOL)
    
    #print data
    f.close()
    
def normalize_data_x(data_x,sums_x,name):
    for idx in xrange(len(sums_x)):
        if sums_x[idx] == 0.:
            print 'input for '+name+' user_id %i has all elements zero will not attempt to normalize '%idx
            sums_x[idx] = 1
    
    return (data_x.transpose()/sums_x).transpose()

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()

    cfg = LoadConfig(options.config)
    validate_config(cfg)
    
    if (cfg.input.csv_contains_counts):
        #read in csv and normalize
        raw_counts = read_user_word_counts(cfg)
    
    if (cfg.input.csv_contains_pixels):
        #read in pixel values between 0 and 1
        raw_counts = read_user_word_pixels(cfg)
    
    if (raw_counts == None):
        print "You need to set either cfg.input.csv_contains_pixels or cfg.input.csv_contains_counts to True otherwise not sure how to process data"
    
    render_file_has_labels = cfg.input.render_data_has_labels
    
    user_labels = None
    if (render_file_has_labels):
        user_labels = read_user_labels(cfg)
        
    #output into pickled data files
    normalize_and_output_pickled_data(cfg,raw_counts, user_labels)
    
