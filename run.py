import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import csv
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from struct import *
from hashmapd import *

#################################
## stuff relating to running the SMH step
#################################

def load_data_without_labels(dataset):
    print '... loading render data, without labels'

    f = gzip.open(dataset,'rb')
    data_x = cPickle.load(f)
    f.close()

    print data_x.shape

    #not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    data_x_shared  = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    
    return data_x_shared

def load_data_with_labels(dataset):

    # unlike the train_SMH case we expect both unsupervised data and appropriate labels for that data, in pairs,
    print '... loading render data, expecting input and labels in pairs'

    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    train_set_x, train_set_labels = train_set
    
    #not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    train_set_x_shared  = theano.shared(numpy.asarray(train_set_x[0:500], dtype=theano.config.floatX))
    train_set_labels_shared  = theano.shared(numpy.asarray(train_set_labels[0:500], dtype=theano.config.floatX))
    
    return [train_set_x_shared, train_set_labels_shared]


def save_model(smh, weights_file='data/last_smh_model_params.pkl.gz'):
    save_file=open(weights_file,'wb')
    cPickle.dump(smh.exportModel(), save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(n_ins=784,  mid_layer_sizes = [200],
               inner_code_length = 10, weights_file='data/last_smh_model_params.pkl.gz'):
    
    numpy_rng = numpy.random.RandomState(212)
    smh = SMH(numpy_rng = numpy_rng,  mid_layer_sizes = mid_layer_sizes, inner_code_length = inner_code_length, n_ins = n_ins)
    save_file=open(weights_file)
    smh_params = cPickle.load(save_file)
    save_file.close()
    smh.loadModel(smh_params)
    return smh

def output_trace(smh, data_x, prefix="run", skip_trace_images=True):
    
    if skip_trace_images:
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

#fixed filename in binary makes for non-scalable code (only one instance can run at a time) ;-(
TSNE_OUTPUT_FILE = "result.dat"; #turns out it *has* to be this to play nicely with the c++ binary 
PCA_INTERIM_FILE = "data.dat"; #turns out it *has* to be this to play nicely with the c++ binary 


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
    
def write_csv_coords(coords, output_file="out/coords.csv"):
    #coords = scale_to_interval(coords, max=100)
    
    print 'writing coordinates to csv'
    csv_writer = csv.writer(open(output_file, 'wb'), delimiter=',')
    for row in coords:
        csv_writer.writerow(["%.10f"%code for code in row]) # format with 10dp accuracy (but no '-e' format stuff)


def write_csv_labels(labels, output_file="out/labels.csv"):
    
    print 'writing labels to csv'
    csv_writer = csv.writer(open(output_file, 'wb'), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for label in labels:
        csv_writer.writerow(("%d"%label,)) 


def write_csv_codes(codes, output_file = "out/codes.csv"):
    
    print 'writing output codes to csv'
    csv_writer = csv.writer(open(output_file, 'wb'), delimiter=',')
    for row_id in xrange(len(codes)):
        row = []
        row.append("%i"%row_id)
        for code in codes[row_id]:
            row.append("%.10f"%code)# format with 10dp accuracy (but no '-e' format stuff)
        csv_writer.writerow(row) 


def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])
    #validate_config(cfg)

    render_file = cfg.input.render_data #NB includes labels or sometimes not?
    render_file_has_labels = cfg.input.render_data_has_labels
    weights_file = cfg.train.weights_file
    n_ins = cfg.shape.input_vector_length
    skip_trace_images = cfg.train.skip_trace_images
    
    input_vector_length = cfg.shape.input_vector_length
    mid_layer_sizes = list(cfg.shape.mid_layer_sizes)
    inner_code_length = cfg.shape.inner_code_length
    
    coords_file = cfg.output.coords_file
    labels_file = cfg.output.labels_file
    codes_file = cfg.output.codes_file
    
    desired_dims = cfg.tsne.desired_dims; # almost always 2
    pca_dims = cfg.tsne.pca_dims; #set this to inner_code_length to effectively skip the PCA step
    perplexity = cfg.tsne.perplexity; #roughly 'the optimal number of neighbours'
    
    # load weights file and initialize smh
    if (render_file_has_labels):
        dataset_x, dataset_labels = load_data_with_labels(dataset=render_file)
    else:
        dataset_x = load_data_without_labels(dataset=render_file)
    
    smh = load_model(n_ins=input_vector_length,  mid_layer_sizes = mid_layer_sizes,
                    inner_code_length = inner_code_length, weights_file=weights_file)
    
    # check it is setup right by reconstructing the input
    output_trace(smh, dataset_x, skip_trace_images)
    
    # run the input data forward through the smh
    codes = get_output_codes(smh, dataset_x)
    
    #output codes
    write_csv_codes(codes, codes_file)
    
    # run the middle layer 'semantic hashes' or 'codes' through Stochastic Neighbour Embedding library
    coords = calc_tsne(codes, desired_dims, perplexity, pca_dims);
    
    # write results and labels
    write_csv_coords(coords, coords_file)
    
    if (render_file_has_labels):
        write_csv_labels(labels, labels_file)
        
if __name__ == '__main__':
    sys.exit(main())
    
    