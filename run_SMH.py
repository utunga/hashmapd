import os, sys

import numpy, time, cPickle, gzip, PIL.Image
import csv
import theano
import theano.tensor as T
from SMH import SMH
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
from struct import *

#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#truncatd from full set for easier test
TRUNCATED_MNIST_FILE = "data/truncated_mnist.pkl.gz"
UNSUPERVISED_MNIST_WEIGHTS_FILE = "data/unsuperivsed_mnist_weights.pkl.gz"
NUM_PIXELS = 784;
SKIP_TRACE = False

#WORD_VECTORS_FILE = "data/word_vectors.pkl.gz";
VECTORS_FILE_DISPLAY = "data/word_vectors_display.pkl.gz";
WORD_VECTORS_WEIGHTS_FILE = "data/word_vectors_weights.pkl.gz"
WORD_VECTORS_NUM_WORDS = 3000; #number of different words in above file ('word_id' column is allowed to be *up to and including* this number)


#################################
## stuff relating to running the SMH step
#################################

def load_unsupervised_data(dataset):
    print '... loading data'

    f = gzip.open(dataset,'rb')
    data_x = cPickle.load(f)
    f.close()

    #need to use train_set since this is where all the data is
    print data_x.shape
    #not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    data_x_shared  = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    
    return data_x_shared

def load_mnist_data(dataset):
    # unlike the train_SMH case we expect both unsupervised data and appropriate labels for that data, in pairs,
    # but we only use the test data.. so only return that 
    print '... loading data'

    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    #test_set_x, test_set_labels = test_set
    #
    ##not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    #test_set_x_shared  = theano.shared(numpy.asarray(test_set_x[0:500], dtype=theano.config.floatX))
    #test_set_labels_shared  = theano.shared(numpy.asarray(test_set_labels[0:500], dtype=theano.config.floatX))
    #
    #return [test_set_x_shared, test_set_labels_shared]
    
    train_set_x, train_set_labels = train_set
    
    #not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    train_set_x_shared  = theano.shared(numpy.asarray(train_set_x[0:500], dtype=theano.config.floatX))
    train_set_labels_shared  = theano.shared(numpy.asarray(train_set_labels[0:500], dtype=theano.config.floatX))
    
    return [train_set_x_shared, train_set_labels_shared]

DEFAULT_WEIGHTS_FILE='data/last_smh_model_params.pkl.gz'
def save_model(smh, weights_file=DEFAULT_WEIGHTS_FILE):
    save_file=open(weights_file,'wb')
    cPickle.dump(smh.exportModel(), save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(n_ins=784,  mid_layer_sizes = [200],
                    inner_code_length = 10, weights_file=DEFAULT_WEIGHTS_FILE):
    numpy_rng = numpy.random.RandomState(212)
    smh = SMH(numpy_rng = numpy_rng,  mid_layer_sizes = mid_layer_sizes, inner_code_length = inner_code_length, n_ins = n_ins)
    save_file=open(weights_file)
    smh_params = cPickle.load(save_file)
    save_file.close()
    smh.loadModel(smh_params)
    return smh

def output_trace(smh, data_x, prefix="run"):
    
    if SKIP_TRACE:
        return
    
    output_y = smh.output_given_x(data_x.value);
    
    # Plot image and reconstrution 
    image = PIL.Image.fromarray(tile_raster_images( X = data_x.value,
             img_shape = (28,28),tile_shape = (10,10), 
             tile_spacing=(1,1)))
    image.save('trace/%s_input.png'%prefix)
    
    image = PIL.Image.fromarray(tile_raster_images( X = output_y,
             img_shape = (28,28),tile_shape = (10,10), 
             tile_spacing=(1,1)))
    image.save('trace/%s_reconstruction.png'%prefix)

def get_output_codes(smh, data_x):
    
    print 'running input data forward through smh..'
    output_codes = smh.output_codes_given_x(data_x.value);
    return output_codes; #a 2d array consisting of 'smh' representation of each input row as a row of floats

#################################
## stuff relating to running the t-sne step
#################################

COORDS_OUTPUT_FILE = "out/coords.csv";
LABELS_OUTPUT_FILE = "out/labels.csv"
TSNE_OUTPUT_FILE = "result.dat"; #turns out it *has* to be this to play nicely with the c++ binary 
PCA_INTERIM_FILE = "data.dat"; #turns out it *has* to be this to play nicely with the c++ binary #tsne_pca_interim.dat";


def calc_tsne(data_matrix,NO_DIMS=2,PERPLEX=30,INITIAL_DIMS=30,LANDMARKS=1):
    """
    This is the main tsne function. In process of doing this writes, then clears up interim file(s)
    data_matrix is a 2D numpy array containing your data (each row is a data point)
    Remark : LANDMARKS is a ratio (0<LANDMARKS<=1)
    If LANDMARKS == 1 , it returns the list of points in the same order as the input
    """
    
    
    #data_matrix=do_pca(data_matrix,INITIAL_DIMS)
    write_data(data_matrix,NO_DIMS,PERPLEX,LANDMARKS)
    do_tSNE()
    Xmat,LM,costs=read_results()
    clear_interim_data()
    if LANDMARKS==1:
        X=re_order(Xmat,LM)
        return X
    return Xmat,LM

def do_tSNE():
    """
    Calls the tsne c++ implementation depending on the platform
    """
    platform=sys.platform
    print'Platform detected : %s'%platform
    if platform in ['mac', 'darwin'] :
        cmd='lib/tSNE_maci'
    elif platform == 'win32' :
        cmd='lib/tSNE_win'
    elif platform == 'linux2' :
        cmd='lib/tSNE_linux'
    else :
        print 'Not sure about the platform, we will try linux version...'
        cmd='../lib/tSNE_linux'
    print 'Calling executable "%s"'%cmd
    os.system(cmd)
    
def do_pca(data_matrix, INITIAL_DIMS) :
    """
    Performs PCA on data.
    Reduces the dimensionality to INITIAL_DIMS
    """
    print 'Performing PCA'

    data_matrix= data_matrix-data_matrix.mean(axis=0)

    if data_matrix.shape[1]<INITIAL_DIMS:
        INITIAL_DIMS=data_matrix.shape[1]

    (eigValues,eigVectors)=numpy.linalg.eig(numpy.cov(data_matrix.T))
    perm=numpy.argsort(-eigValues)
    eigVectors=eigVectors[:,perm[0:INITIAL_DIMS]]
    data_matrix=numpy.dot(data_matrix,eigVectors)
    return data_matrix

def read_bin(type,file) :
    """
    used to read binary data from a file
    """
    return unpack(type,file.read(calcsize(type)))

def read_results():
    """
    Reads result from result.dat
    """
    print 'Reading tsne results..'
    f=open(TSNE_OUTPUT_FILE,'rb')
    n,ND=read_bin('ii',f)
    Xmat=numpy.empty((n,ND))
    for i in range(n):
        for j in range(ND):
            Xmat[i,j]=read_bin('d',f)[0]
    LM=read_bin('%ii'%n,f)
    costs=read_bin('%id'%n,f)
    f.close()
    return (Xmat,LM,costs)

def re_order(Xmat, LM):
    """
    Re-order the data in the original order
    Call only if LANDMARKS==1
    """
    print 'Reordering results'
    X=numpy.zeros(Xmat.shape)
    for i,lm in enumerate(LM):
        X[lm]=Xmat[i]
    return X

def clear_interim_data():
    """
    Clears files data.dat and result.dat
    """
    print 'Clearing interim data files'
    os.system('rm %s'%PCA_INTERIM_FILE)
    os.system('rm %s'%TSNE_OUTPUT_FILE)

def write_data(data_matrix,NO_DIMS,PERPLEX,LANDMARKS):
    print 'Writing data.dat'
    print 'Dimension of projection : %i \nPerplexity : %i \nLandmarks(ratio) : %f'%(NO_DIMS,PERPLEX,LANDMARKS)
    n,d = data_matrix.shape
    f = open('data.dat', 'wb')
    f.write(pack('=iiid',n,d,NO_DIMS,PERPLEX))
    f.write(pack('=d',LANDMARKS))
    for inst in data_matrix :
        for el in inst :
            f.write(pack('=d',el))
    f.close()

def scale_to_interval(arr,max=1.0, eps=1e-8):
    """ Scales all values in the numpy array to be between 0 and max """
    
    print 'scaling to interval [0,%s]'%max
    orig_shape = arr.shape
    flatt = arr.ravel() #flatten and then scale
    flatt -= flatt.min()
    flatt *= max/ (flatt.max()+eps)
    return flatt.reshape(orig_shape)
    
def write_csv_coords(coords, labels):
    #coords = scale_to_interval(coords, max=100)
    
    print 'writing coordinates to csv'
    csv_writer = csv.writer(open(COORDS_OUTPUT_FILE, 'wb'), delimiter=',')
    for row in coords:
        csv_writer.writerow(["%.10f"%code for code in row]) # format with 10dp accuracy (but no '-e' format stuff)

    if labels==None:
        return
    print 'writing labels to csv'
    csv_writer = csv.writer(open(LABELS_OUTPUT_FILE, 'wb'), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for label in labels:
        csv_writer.writerow(("%d"%label,)) 

if __name__ == '__main__':
   
    data_file = TRUNCATED_MNIST_FILE #NB includes labels
    weights_file = UNSUPERVISED_MNIST_WEIGHTS_FILE
    n_ins = NUM_PIXELS
    dataset_x, dataset_labels = load_mnist_data(dataset=data_file)
    
    #data_file = VECTORS_FILE_DISPLAY
    #weights_file = WORD_VECTORS_WEIGHTS_FILE
    #n_ins = WORD_VECTORS_NUM_WORDS
    #dataset_x = load_unsupervised_data(dataset=data_file)
    #dataset_labels = None
    #SKIP_TRACE = True
    
    # load weights file and initialize smh
    
    smh = load_model(n_ins=n_ins,  mid_layer_sizes = [400,200],
                    inner_code_length = 30, weights_file=weights_file)
    
    # check it is setup right by reconstructing the input
    output_trace(smh, dataset_x)
    
    # run the input data forward through the smh
    codes = get_output_codes(smh, dataset_x)
    
    # run the middle layer 'semantic hashes' or 'codes' through Stochastic Neighbour Embedding library
    desired_dims = 2;
    pca_dims = 30; #don't shrink through the PCA step at all
    perplexity = 10;
    coords = calc_tsne(codes, desired_dims, perplexity, pca_dims);
    
    # write results and labels
    write_csv_coords(coords, dataset_labels)