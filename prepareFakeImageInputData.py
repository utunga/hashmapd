from struct import *
from numpy import *
import sys
import os
import csv
import cPickle
import gzip
import theano
import PIL.Image
from hashmapd.utils import tile_raster_images

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
WORD_VECTORS_FILE = "data/fake_img_data.csv";
PICKLED_WORD_VECTORS_FILE = "data/fake_img_data.pkl.gz";
WORD_VECTORS_NUM_WORDS = 784 
WORD_VECTORS_NUM_USERS = 14


def read_user_word_pixels():
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + WORD_VECTORS_FILE
    
    raw_pixels = zeros((WORD_VECTORS_NUM_USERS, WORD_VECTORS_NUM_WORDS), dtype=theano.config.floatX);#store as floa64 so that normalized_pixels uses float math
    
    vectorReader = csv.DictReader(open(WORD_VECTORS_FILE, 'rb'), delimiter=',')
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
    #print normalized_pixels;    
    #return normalized_pixels;

def output_pickled_data(raw_pixels):
    
   
    print "outputting full data set"
    train_set_x = raw_pixels
    valid_set_x = raw_pixels
    test_set_x = raw_pixels
   
    print '...  pickling and zipping data to '+ PICKLED_WORD_VECTORS_FILE
    f = gzip.open(PICKLED_WORD_VECTORS_FILE,'wb')
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
    
  
    print(train_set_x.value.shape)
    
    # Plot image and reconstrution 
    image = PIL.Image.fromarray(tile_raster_images( X = train_set_x.value,
             img_shape = (28,28),tile_shape = (14,1), 
             tile_spacing=(1,1)))
    image.save('trace/fake_input.png')
        


if __name__ == '__main__':
    raw_pixels = read_user_word_pixels();
    output_pickled_data(raw_pixels);
    test_pickling()