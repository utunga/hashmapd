
from struct import *
from numpy import *
import sys
import getopt
import os
import csv
import cPickle
import gzip
import theano
import time, PIL.Image

from hashmapd import *

PICKLED_WORD_VECTORS_TRAINING_FILE_POSTFIX = "training_data";
PICKLED_WORD_VECTORS_VALIDATION_FILE_POSTFIX = "validation_data";
PICKLED_WORD_VECTORS_TESTING_FILE_POSTFIX = "testing_data";
PICKLED_FILE_TYPE = ".pkl.gz"

def read_user_word_pixels(cfg):
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + cfg.input.csv_data
    
    raw_pixels = zeros((cfg.input.number_of_examples, cfg.shape.input_vector_length), dtype=theano.config.floatX); #store as float so that normalized_counts uses float math
    
    vectorReader = csv.DictReader(open(cfg.input.csv_data, 'rb'), delimiter=',')
    iter=0;
    for row in vectorReader:
        if iter % 100==0:
            print 'reading row '+ str(iter) + '..'; #MKT: presumably a nicer way to do this
        iter += 1;
        user_id = int(row['user_id']);
        word_id = int(row['word_id']);
        pixel = float(row['pixel']);
        raw_pixels[user_id,word_id] = pixel;
    
    #total_user_pixels = raw_pixels.sum(axis=1)
    #normalized_pixels = (raw_pixels.transpose()/total_user_pixels).transpose();
    
    print 'done reading input';
    print raw_pixels;
    return raw_pixels;
    
def read_user_word_counts(cfg):
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + cfg.input.csv_data
    
    raw_counts = zeros((cfg.input.number_of_examples, cfg.shape.input_vector_length), dtype=theano.config.floatX); #store as float so that normalized_counts uses float math
    vectorReader = csv.DictReader(open(cfg.input.csv_data, 'rb'), delimiter=',')
    iter=0;
    for row in vectorReader:
        if iter % 10000==0:  #MKT: presumably a nicer way to do this ?
            print 'reading row '+ str(iter) + '..';
        iter += 1;
        user_id = int(row['user_id'])-1;
        word_id = int(row['word_id'])-1;
        count = int(row['count']);
        raw_counts[user_id,word_id] = count;
    
    print 'done reading input';
    return raw_counts;

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
    
    
def normalize_and_output_pickled_data(cfg, raw_counts):
   
    print "outputting full data set"
    
    num_examples = cfg.input.number_of_examples
    train_cutoff = cfg.input.number_for_training
    validate_cutoff = train_cutoff+cfg.input.number_for_validation
    test_cutoff = validate_cutoff+cfg.input.number_for_testing
    batch_size = cfg.train.train_batch_size;
    
    train_set_x = raw_counts[0:train_cutoff]
    train_sums = train_set_x.sum(axis=1);
    mean_doc_size = train_sums.mean();
    
    if (cfg.input.number_for_validation ==0):
        print 'WARNING: no examples set aside for validation, copying train set data for validation (as a quick hack only)'
        valid_set_x = train_set_x
    else:
        valid_set_x = raw_counts[train_cutoff:validate_cutoff]
    valid_sums = valid_set_x.sum(axis=1);

    if (cfg.input.number_for_testing ==0):
        print 'WARNING: no examples set aside for testing, copying validation set data for training (as a quick hack only)'
        test_set_x = valid_set_x
    else:
        test_set_x = raw_counts[validate_cutoff:test_cutoff]
    test_sums = test_set_x.sum(axis=1);
   
    print '...  pickling and zipping train/validate/test data to '+ cfg.input.train_data
    
    train_file = gzip.open(cfg.input.train_data+PICKLED_WORD_VECTORS_TRAINING_FILE_POSTFIX+'0'+PICKLED_FILE_TYPE,'wb')
    valid_file = gzip.open(cfg.input.train_data+PICKLED_WORD_VECTORS_VALIDATION_FILE_POSTFIX+'0'+PICKLED_FILE_TYPE,'wb')
    test_file = gzip.open(cfg.input.train_data+PICKLED_WORD_VECTORS_TESTING_FILE_POSTFIX+'0'+PICKLED_FILE_TYPE,'wb')
    
    if (cfg.train.first_layer_type=='poisson'):
        cPickle.dump((train_set_x,train_sums,zeros(train_sums.shape,dtype=theano.config.floatX)), train_file, cPickle.HIGHEST_PROTOCOL);
        cPickle.dump((valid_set_x,valid_sums,zeros(valid_sums.shape,dtype=theano.config.floatX)), valid_file, cPickle.HIGHEST_PROTOCOL);
        cPickle.dump((test_set_x,test_sums,zeros(test_sums.shape,dtype=theano.config.floatX)), test_file, cPickle.HIGHEST_PROTOCOL);
    else:
        cPickle.dump((normalize_data_x(train_set_x,train_sums,'training'),train_sums,zeros(train_sums.shape,dtype=theano.config.floatX)), train_file, cPickle.HIGHEST_PROTOCOL);
        cPickle.dump((normalize_data_x(valid_set_x,valid_sums,'validation'),valid_sums,zeros(valid_sums.shape,dtype=theano.config.floatX)), valid_file, cPickle.HIGHEST_PROTOCOL);
        cPickle.dump((normalize_data_x(test_set_x,test_sums,'testing'),test_sums,zeros(test_sums.shape,dtype=theano.config.floatX)), test_file, cPickle.HIGHEST_PROTOCOL);
    
    train_file.close();
    valid_file.close();
    test_file.close(); 
        
    f = gzip.open('data/word_vectors_info.pkl.gz','wb');
    cPickle.dump(("data/word_vectors_"+PICKLED_WORD_VECTORS_TRAINING_FILE_POSTFIX,1,train_cutoff/batch_size,
                    "data/word_vectors_"+PICKLED_WORD_VECTORS_VALIDATION_FILE_POSTFIX,1,validate_cutoff/batch_size,
                    "data/word_vectors_"+PICKLED_WORD_VECTORS_TESTING_FILE_POSTFIX,1,test_cutoff/batch_size,
                    (train_cutoff+validate_cutoff+test_cutoff)/batch_size,mean_doc_size), f, cPickle.HIGHEST_PROTOCOL);
    f.close();

    print '...  pickling and zipping render_data to '+ cfg.input.render_data
    render_data = normalize_data_x(train_set_x,train_sums,'training')[0:num_examples]
    f = gzip.open(cfg.input.render_data,'wb')
    cPickle.dump(render_data,f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    
def normalize_data_x(data_x,sums_x,name):
    for idx in xrange(len(sums_x)):
        if sums_x[idx] == 0.:
            print 'input for '+name+' user_id %i has all elements zero will not attempt to normalize '%idx
            sums_x[idx] = 1
    
    return (data_x.transpose()/sums_x).transpose();

def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])
    validate_config(cfg)
    
    if (cfg.input.csv_contains_counts):
        #read in csv and normalize
        raw_counts = read_user_word_counts(cfg);
    
    if (cfg.input.csv_contains_pixels):
        #read in pixel values between 0 and 1
        raw_counts = read_user_word_pixels(cfg);
    
    
    if (raw_counts == None):
        print "You need to set either cfg.input.csv_contains_pixels or cfg.input.csv_contains_counts to True otherwise not sure how to process data"
    
    #output into pickled data files
    normalize_and_output_pickled_data(cfg,raw_counts);

if __name__ == '__main__':
    sys.exit(main())    
    