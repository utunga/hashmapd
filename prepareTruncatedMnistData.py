
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
UNSUPERVISED_MNIST = 'data/truncated_unsupervised_mnist.pkl.gz'

    
def load_and_truncate_mnist():
 
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
    train_set_x = train_set_x[0:5000]
    valid_set_x = valid_set_x[0:1000]
    test_set_x = test_set_x[0:1000]
        
    print '...  pickling and zipping truncated, unsupervised data to '+ UNSUPERVISED_MNIST
    f = gzip.open(UNSUPERVISED_MNIST,'wb')
    cPickle.dump((train_set_x, valid_set_x, test_set_x),f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    

def test_truncated_mnist():
    f = gzip.open(UNSUPERVISED_MNIST,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
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
    
    load_and_truncate_mnist()
    test_truncated_mnist()
   