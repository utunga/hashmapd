from struct import *
from numpy import *
import sys
import os
import csv
import cPickle
import gzip
import theano

# Some fileconstants

##real data
#WORD_VECTORS_FILE = "data/user_word_vectors.csv";
#PICKLED_WORD_VECTORS_FILE = "data/word_vectors.pkl.gz";
#TEST_PICKLED_WORD_VECTORS_FILE = "data/test_word_vectors.pkl.gz"
#WORD_VECTORS_NUM_WORDS = 3000; #number of different words in above file ('word_id' column is allowed to be *up to and including* this number)
#WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)

#test data
#WORD_VECTORS_FILE = "data/test_word_vectors.csv";
#WORD_VECTORS_NUM_WORDS = 500; #500 words for testing.. (speeds stuff up a bit)
#WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)

#fake data
WORD_VECTORS_FILE = "data/fake_word_vectors.csv";
PICKLED_WORD_VECTORS_FILE = "data/fake_word_vectors.pkl.gz";
FAKE_WORD_VECTORS_NUM_WORDS = 50; 
FAKE_WORD_VECTORS_NUM_USERS = 20;


def read_user_word_counts():
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + WORD_VECTORS_FILE
    
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
    normalized_counts = (raw_counts.transpose()/total_user_counts).transpose();
    
    print 'done reading input';
    print raw_counts;
    print normalized_counts;    
    return normalized_counts;

def output_pickled_data(normalized_counts):
    
   
    print "outputting full data set"
    train_set_x = normalized_counts[0:600]
    valid_set_x = normalized_counts[600:650]
    test_set_x = normalized_counts[650:WORD_VECTORS_NUM_USERS]
   
    print '...  pickling and zipping data to '+ PICKLED_WORD_VECTORS_FILE
    f = gzip.open(PICKLED_WORD_VECTORS_FILE,'wb')
    cPickle.dump((train_set_x, valid_set_x, test_set_x),f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    
    print '...  truncating to smaller test set'
    #truncate the data so it fits in the damn gpu
    train_set_x = normalized_counts[0:200]
    valid_set_x = normalized_counts[200:250]
    test_set_x = normalized_counts[250:350]
   
    print '...  pickling and zipping truncated data to '+ TEST_PICKLED_WORD_VECTORS_FILE
    f = gzip.open(TEST_PICKLED_WORD_VECTORS_FILE,'wb')
    cPickle.dump((train_set_x, valid_set_x, test_set_x),f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    

def test_pickling(dataset=PICKLED_WORD_VECTORS_FILE):

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
        shared_x = theano.shared(asarray(data_x, dtype=theano.config.floatX))
        return shared_x
    
    test_set_x  = shared_dataset(test_set)
    valid_set_x = shared_dataset(valid_set)
    train_set_x = shared_dataset(train_set)


if __name__ == '__main__':
    normalized_counts = read_user_word_counts();
    output_pickled_data(normalized_counts);
    test_pickling()