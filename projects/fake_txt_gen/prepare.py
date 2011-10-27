import sys
import getopt
import os
import csv
import cPickle
import gzip
import theano
import numpy as np
import numpy.lib.format

from hashmapd.load_config import LoadConfig

TRAINING_FILE = "training_data"
VALIDATION_FILE = "validation_data"
TESTING_FILE = "testing_data"
RENDER_FILE = 'render_data'
PICKLED_TYPE = ".pkl.gz"
        
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
    print "reading labels from" + cfg.raw.users_file
    
    users = []
    vectorReader = csv.DictReader(open(cfg.raw.users_file, 'rb'), delimiter=',')
    for (i, row) in enumerate(vectorReader):
        user_id = int(row['user_id'])
        label = row['screen']
        assert user_id == i, (user_id, i)
        users.append(label)
    return users

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
    

def normalize_and_output_pickled_data(cfg, raw_counts, user_labels):
   
    print "outputting full data set"
    
    num_examples = cfg.input.number_of_examples
    train_cutoff = cfg.input.number_for_training
    validate_cutoff = train_cutoff+cfg.input.number_for_validation
    test_cutoff = validate_cutoff+cfg.input.number_for_testing
    
    print raw_counts
    train_set_x = raw_counts[0:train_cutoff]
    user_labels = user_labels[0:train_cutoff]
    
    if (cfg.input.number_for_validation ==0):
        print 'WARNING: no examples set aside for validation, copying train set data for validation (as a quick hack only)'
        valid_set_x = train_set_x
    else:
        valid_set_x = raw_counts[train_cutoff:validate_cutoff]

    if (cfg.input.number_for_testing ==0):
        print 'WARNING: no examples set aside for testing, copying validation set data for training (as a quick hack only)'
        test_set_x = valid_set_x
    else:
        test_set_x = raw_counts[validate_cutoff:test_cutoff]
   
    print '...  writing train, validate, test and render data to data directory'
    
    for (filename, x, y) in [
            (TRAINING_FILE, train_set_x, None),
            (VALIDATION_FILE, valid_set_x, None),
            (TESTING_FILE, test_set_x, None),
            (RENDER_FILE, train_set_x, user_labels)]:
        if cfg.train.first_layer_type != 'poisson':  
            x = normalize_data_x(x, filename)
        data = (x, None, y)
        filename = os.path.join("data", filename+'_0.pkl.gz')
        cPickle.dump(data, gzip.open(filename, 'wb'), cPickle.HIGHEST_PROTOCOL)
            
def normalize_data_x(data_x, name):
    sums_x = data_x.sum(axis=1)[:, numpy.newaxis]
    if any(sums_x == 0):
        print 'Some input has all elements zero, will not attempt to normalize it'
        sums_x[sums_x == 0] = 1 
    return data_x / sums_x

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()

    cfg = LoadConfig(options.config)
    validate_config(cfg)

    raw_counts = read_user_word_counts(cfg)
    user_labels = read_user_labels(cfg)
        
    #output into pickled data files
    normalize_and_output_pickled_data(cfg, raw_counts, user_labels)
    
