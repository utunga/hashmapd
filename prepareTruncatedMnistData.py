from hashmapd import *

from struct import *
from numpy import *
import sys
import os
import csv
import cPickle
import gzip
import theano
import time, PIL.Image

from hashmapd.utils import tile_raster_images
# Some fileconstants

##real data

MNIST_FILE = 'data/mnist.pkl.gz'
OUTPUT_FOLDER = 'data/mnist/'

    
def load_and_truncate_mnist(batch_size):
 
    full_dataset_file=MNIST_FILE
    print '...  loading full data from '+ full_dataset_file
    f = gzip.open(full_dataset_file,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    train_set_x = train_set[0]
    valid_set_x = valid_set[0]
    test_set_x = test_set[0]
    
    # just discard these, only useful for supervised training
    # train_set_y = train_set[1]
    # valid_set_y = valid_set[1]
    # test_set_y = test_set[1]
    
    
    print '...  truncating to smaller set'
    #truncate the data so it fits in the damn gpu
    train_set_x = test_set_x[0:5000]
    valid_set_x = valid_set_x[0:1000]
    test_set_x = train_set_x[0:1000]
        
    print '...  pickling and zipping truncated, unsupervised data to '+ OUTPUT_FOLDER
    
    # dump in required format
    f = gzip.open(OUTPUT_FOLDER+'mnist_data_info.pkl.gz','wb')
    cPickle.dump((OUTPUT_FOLDER+'mnist_training_data',1,int(math.floor(5000/batch_size)),
                OUTPUT_FOLDER+'mnist_validation_data',1,int(math.floor(1000/batch_size)),
                OUTPUT_FOLDER+'mnist_testing_data',1,int(math.floor(1000/batch_size)),
                int(math.floor(5000/batch_size)),1),f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    f = gzip.open(OUTPUT_FOLDER+'mnist_training_data0.pkl.gz','wb')
    cPickle.dump((train_set_x,[0]*5000,[]),f, cPickle.HIGHEST_PROTOCOL) # no sums/labels
    f.close()
    
    f = gzip.open(OUTPUT_FOLDER+'mnist_validation_data0.pkl.gz','wb')
    cPickle.dump((valid_set_x,[0]*5000,[]),f, cPickle.HIGHEST_PROTOCOL) # no sums/labels
    f.close()
    
    f = gzip.open(OUTPUT_FOLDER+'mnist_testing_data0.pkl.gz','wb')
    cPickle.dump((test_set_x,[0]*5000,[]),f, cPickle.HIGHEST_PROTOCOL) # no sums/labels
    f.close()
    

def test_truncated_mnist():
    f = gzip.open(OUTPUT_FOLDER+'mnist_training_data0.pkl.gz','rb')
    train_set,sums,labels = cPickle.load(f)
    f.close()
    
    # Plot filters after each training epoch
    
    # Construct image from the weight matrix 
    image = PIL.Image.fromarray(tile_raster_images( train_set,
             img_shape = (28,28),tile_shape = (30,30), 
             tile_spacing=(1,1)))
    image.save('truncated_input.png')
    
    f = gzip.open(MNIST_FILE,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
if __name__ == '__main__':
    
    cfg = LoadConfig("unsupervised_mnist")
    
    load_and_truncate_mnist(cfg.train.train_batch_size)
    test_truncated_mnist()
   