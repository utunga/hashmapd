
from struct import *
from numpy import *
import sys
import os
import csv
import cPickle
import gzip
import theano
import time, PIL.Image

from utils import tile_raster_images
# Some fileconstants

##real data

WORD_VECTORS_FILE = "data/user_word_vectors.csv";
PICKLED_WORD_VECTORS_FILE = "data/word_vectors.pkl.gz";
PICKLED_VECTORS_FILE_DISPLAY = "data/word_vectors_display.pkl.gz";
WORD_VECTORS_NUM_WORDS = 3000; #number of different words in above file ('word_id' column is allowed to be *up to and including* this number)
WORD_VECTORS_NUM_USERS = 5285; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)


def read_user_word_counts():
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + WORD_VECTORS_FILE
    
    #vectorReader = csv.DictReader(open(WORD_VECTORS_FILE, 'rb'), delimiter=',')
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
    #raw_counts = zeros((max_user_id+1, max_word_id+1), dtype=float64);#store as floa64 so that normalized_counts uses float math

    raw_counts = zeros((WORD_VECTORS_NUM_USERS, WORD_VECTORS_NUM_WORDS), dtype=float64);#store as floa64 so that normalized_counts uses float math
    
    vectorReader = csv.DictReader(open(WORD_VECTORS_FILE, 'rb'), delimiter=',')
    iter=0;
    for row in vectorReader:
        if iter % 10000==0:
            print 'reading row '+ str(iter) + '..'; #MKT: presumably a nicer way to do this
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
    #print raw_counts;
    #print normalized_counts;    
    return normalized_counts;

def output_pickled_data(normalized_counts):
   
    print "outputting full data set"
    train_set_x = normalized_counts[0:4000]
    valid_set_x = normalized_counts[4000:4500]
    test_set_x = normalized_counts[4500:WORD_VECTORS_NUM_USERS]
   
    print '...  pickling and zipping data to '+ PICKLED_WORD_VECTORS_FILE
    f = gzip.open(PICKLED_WORD_VECTORS_FILE,'wb')
    cPickle.dump((train_set_x, valid_set_x, test_set_x),f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    #print '...  pickling and zipping data to '+ PICKLED_VECTORS_FILE_DISPLAY
    #all_words = eye(WORD_VECTORS_NUM_WORDS, dtype=float64);
    #data_x = concatenate((train_set_x,valid_set_x,test_set_x, all_words),axis=0)
    #f = gzip.open(PICKLED_VECTORS_FILE_DISPLAY,'wb')
    #cPickle.dump(data_x,f, cPickle.HIGHEST_PROTOCOL)
    #f.close()

    print '...  pickling and zipping data to '+ PICKLED_VECTORS_FILE_DISPLAY
    #data_x = concatenate((train_set_x,axis=0)
    all_data = normalized_counts[0:5000]
    f = gzip.open(PICKLED_VECTORS_FILE_DISPLAY,'wb')
    cPickle.dump(all_data,f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    

def normalize_data_x(data_x):
    totals_for_rows = data_x.sum(axis=1)
    normalized_data = (data_x.transpose()/totals_for_rows).transpose();
    return normalized_data;
    
if __name__ == '__main__':
    normalized_counts = read_user_word_counts();
    output_pickled_data(normalized_counts);
    