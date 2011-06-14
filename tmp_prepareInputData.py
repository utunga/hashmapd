
from struct import *
from numpy import *
import sys
import os
import csv
import cPickle
import gzip
import theano
import time

from utils import tiled_array_image
# Some fileconstants

##real data

MNIST_FILE = 'data/truncated_mnist.pkl.gz'
NORMALIZED_MNIST_FILE = 'data/truncated_normalized_mnist.pkl.gz'
WORD_VECTORS_FILE = "data/user_word_vectors.csv";
PICKLED_WORD_VECTORS_FILE = "data/word_vectors.pkl.gz";
#TEST_PICKLED_WORD_VECTORS_FILE = "data/test_word_vectors.pkl.gz"
WORD_VECTORS_NUM_WORDS = 3000; #number of different words in above file ('word_id' column is allowed to be *up to and including* this number)
WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)
NORMALIZED_UNSUPERVISED_MNIST = 'data/truncated_unsupervised_mnist.pkl.gz'


#test data
#WORD_VECTORS_FILE = "data/test_word_vectors.csv";
#WORD_VECTORS_NUM_WORDS = 500; #500 words for testing.. (speeds stuff up a bit)
#WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)

#fake data
#WORD_VECTORS_FILE = "data/fake_word_vectors.csv";
#PICKLED_WORD_VECTORS_FILE = "data/fake_word_vectors.pkl.gz";
#FAKE_WORD_VECTORS_NUM_WORDS = 50; 
#FAKE_WORD_VECTORS_NUM_USERS = 20;


def read_user_word_counts():
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + WORD_VECTORS_FILE
    
    raw_counts = zeros((WORD_VECTORS_NUM_USERS, WORD_VECTORS_NUM_WORDS), dtype=theano.config.floatX);#store as floa64 so that normalized_counts uses float math
    
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
    
def load_and_normalize_mnist(dataset_file=MNIST_FILE):
 
    print '...  loading data from '+ dataset_file
    f = gzip.open(dataset_file,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    train_set_x = train_set[0]
    valid_set_x = valid_set[0]
    test_set_x = test_set[0]
    train_set_y = train_set[1]
    valid_set_y = valid_set[1]
    test_set_y = test_set[1]
    
    
    print '... normalizing data for softmax friendly usage'
    train_set_x = normalize_data_x(train_set_x)
    valid_set_x = normalize_data_x(valid_set_x)
    test_set_x = normalize_data_x(test_set_x)
    
    train_set = (train_set_x, train_set_y)
    test_set = (test_set_x, test_set_y)
    valid_set = (valid_set_x, valid_set_y)

    print '...  pickling and zipping normalized data to '+ NORMALIZED_MNIST_FILE
    f = gzip.open(NORMALIZED_MNIST_FILE,'wb')
    cPickle.dump((train_set, valid_set, test_set),f, cPickle.HIGHEST_PROTOCOL)
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

def test_normalize():
    f = gzip.open(NORMALIZED_MNIST_FILE,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    # Plot filters after each training epoch
    
    # Construct image from the weight matrix 
    image = tiled_array_image( train_set[0])
    image.save('truncated_normalized_input.png')
  
    print train_set[0][3].sum()
    print valid_set[0][3].sum()
    print test_set[0][3].sum()
    print train_set[0][5].sum()
    print valid_set[0][5].sum()
    print test_set[0][5].sum()
    
    f = gzip.open(MNIST_FILE,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    # Plot filters after each training epoch
    
    # Construct image from the weight matrix 
    image = tiled_array_image( train_set[0])
    image.save('truncated_input.png')
    
    print train_set[0][3].sum()
    print valid_set[0][3].sum()
    print test_set[0][3].sum()
    print train_set[0][5].sum()
    print valid_set[0][5].sum()
    print test_set[0][5].sum()
    

def normalize_data_x(data_x):
    totals_for_rows = data_x.sum(axis=1)
    normalized_data = (data_x.transpose()/totals_for_rows).transpose();
    return normalized_data;
    
if __name__ == '__main__':
    normalized_counts = read_user_word_counts();
    output_pickled_data(normalized_counts);
    test_pickling()
    #load_and_normalize_mnist()
    #test_normalize()
   