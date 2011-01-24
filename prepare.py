
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


def read_user_word_counts(cfg):
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + cfg.input.csv_data
    
    #vectorReader = csv.DictReader(open(cfg.input.csv_data, 'rb'), delimiter=',')
    #iter=0;
    #max_word_id = 0
    #max_user_id = 0
    #for row in vectorReader:
    #    if iter % 10000==0:
    #        print 'reading row '+ str(iter) + '..'; #MKT: presumably a nicer way to do this
    #    iter += 1;
    #    user_id = int(row['user_id'])-1;
    #    word_id = int(row['word_id'])-1;
    #    max_word_id = max(word_id, max_word_id)
    #    max_user_id = max(user_id, max_user_id)
    #    
    #print ' max user id %i'%max_user_id
    #print ' max word id %i'%max_word_id
    #
    #raw_counts = zeros((max_user_id+1, max_word_id+1), dtype=floatX);#store as floa64 so that normalized_counts uses float math

    raw_counts = zeros((cfg.input.number_of_examples, cfg.shape.input_vector_length), dtype=theano.config.floatX);#store as floa64 so that normalized_counts uses float math
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
    
    total_user_counts = raw_counts.sum(axis=1)
    for idx in xrange(len(total_user_counts)):
        if total_user_counts[idx] == 0.:
            print 'input for  user_id %i has all elements zero will not attempt to normalize '%idx
            total_user_counts[idx] = 1
            
    normalized_counts = (raw_counts.transpose()/total_user_counts).transpose();
    
    print 'done reading input';
    return normalized_counts;

def validate_config(cfg):
    
    num_examples = cfg.input.number_of_examples
    train_cutoff = cfg.input.number_for_training
    validate_cutoff = train_cutoff+cfg.input.number_for_validation
    test_cutoff = validate_cutoff+cfg.input.number_for_testing
    
    assert test_cutoff<=num_examples, \
        "total of number of examples should be more than cases for train, validate and test"
    assert 0<train_cutoff, \
        "config fail, number_for_training needs to be > 0"
    assert train_cutoff<validate_cutoff, \
        "config fail, number_for_validation should be > 0"
    assert validate_cutoff<test_cutoff, \
        "config fail, number_for_testing should be > 0"
    
    
def output_pickled_data(cfg, normalized_counts):
   
    print "outputing full data set"
    
    num_examples = cfg.input.number_of_examples
    train_cutoff = cfg.input.number_for_training
    validate_cutoff = train_cutoff+cfg.input.number_for_validation
    test_cutoff = validate_cutoff+cfg.input.number_for_testing
    
    train_set_x = normalized_counts[0:train_cutoff]
    valid_set_x = normalized_counts[train_cutoff:validate_cutoff]
    test_set_x = normalized_counts[validate_cutoff:test_cutoff]
   
    print '...  pickling and zipping train/validate/test data to '+ cfg.input.train_data
    f = gzip.open(cfg.input.train_data,'wb')
    cPickle.dump((train_set_x, valid_set_x, test_set_x),f, cPickle.HIGHEST_PROTOCOL)
    f.close()

    print '...  pickling and zipping render_data to '+ cfg.input.render_data
    render_data = normalized_counts[0:num_examples]
    f = gzip.open(cfg.input.render_data,'wb')
    cPickle.dump(render_data,f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    
def normalize_data_x(data_x):
    totals_for_rows = data_x.sum(axis=1)
    normalized_data = (data_x.transpose()/totals_for_rows).transpose();
    return normalized_data;

def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])
    validate_config(cfg)
    
    #read in csv and normalize
    normalized_counts = read_user_word_counts(cfg);
    
    #output into pickled data files
    output_pickled_data(cfg,normalized_counts);

if __name__ == '__main__':
    sys.exit(main())    
    