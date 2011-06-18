import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import csv

def get_git_home():
    testpath = '.'
    while not '.git' in os.listdir(testpath) and not os.path.abspath(testpath) == '/':
        testpath = os.path.sep.join(('..', testpath))
    if not os.path.abspath(testpath) == '/':
        return os.path.abspath(testpath)
    else:
        raise ValueError, "Not in git repository"

HOME = get_git_home()
sys.path.append(HOME)

from hashmapd.load_config import LoadConfig, DefaultConfig
from hashmapd.SMH import SMH

#################################
## stuff relating to running the SMH step
#################################

def load_data_without_labels(dataset):
    print '... loading render data, without labels'

    f = gzip.open(dataset,'rb')
    x = cPickle.load(f)
    f.close()

    print "render data has shape:", x.shape

    return x

    #not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    #x_shared  = theano.shared(numpy.asarray(x, dtype=theano.config.floatX))
    
    #return x_shared
    
def load_data_with_labels(dataset):

    # unlike the train_SMH case we expect both unsupervised data and appropriate labels for that data, in pairs,
    print '... loading render data, expecting input and labels in pairs'

    f = gzip.open(dataset,'rb')
    train_set = cPickle.load(f)
    f.close()
    
    train_set_x, train_set_labels = train_set

    print "render data has shape:", train_set_x.shape
    
    return [train_set_x, train_set_labels]
    
    #not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    #train_set_x_shared  = theano.shared(numpy.asarray(train_set_x[0:500], dtype=theano.config.floatX))
    #train_set_labels_shared  = theano.shared(numpy.asarray(train_set_labels[0:500], dtype=theano.config.floatX))
    
    #return [train_set_x_shared, train_set_labels_shared]
    
def load_data_with_multi_labels(dataset):

    # unlike the train_SMH case we expect both unsupervised data and appropriate labels for that data, in pairs,
    print '... loading render data, expecting input and labels in pairs'

    f = gzip.open(dataset,'rb')
    x, x_sums, labels = cPickle.load(f)
    f.close()
    
    print "render data has shape:", x.shape
    
    # for now, for multi-label data, take the last (most-specific?) label only 
    if labels.shape[0] > 1 and labels.shape[1] > 1:
        concat_labels = []
        for i in xrange(labels.shape[0]):
            concat_labels.append(-1);
            for j in xrange(labels.shape[1]):
                if labels[i,j] == 1:
                    concat_labels[i] = j;
    
    return [x/numpy.array([x_sums]*(x.shape[1])).transpose(),concat_labels]
    
    #not sure if making these things 'shared' helps the GPU out but just in case we may as well do that
    #x_shared  = theano.shared(numpy.asarray(x[0:500], dtype=theano.config.floatX))
    #labels_shared  = theano.shared(numpy.asarray(labels[0:500], dtype=theano.config.floatX))
    
    #return [x_shared, labels_shared]


def save_model(smh, weights_file='data/last_smh_model_params.pkl.gz'):
    save_file=open(weights_file,'wb')
    cPickle.dump(smh.export_model(), save_file, cPickle.HIGHEST_PROTOCOL);
    save_file.close();

def load_model(cost_method, n_ins=784,  mid_layer_sizes = [200],
               inner_code_length = 10, weights_file='data/last_smh_model_params.pkl.gz'):
    
    numpy_rng = numpy.random.RandomState(212)
    smh = SMH(numpy_rng = numpy_rng,  mid_layer_sizes = mid_layer_sizes, inner_code_length = inner_code_length, n_ins = n_ins)
    smh.unroll_layers(cost_method,0) #need to unroll before loading params so that we have right number of layers
    save_file=open(weights_file,'rb')
    smh_params = cPickle.load(save_file)
    save_file.close()
    smh.load_model(smh_params)
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
    
    output_codes = smh.output_codes_given_x(data_x);
    return output_codes; #a 2d array consisting of 'smh' representation of each input row as a row of floats


def scale_to_interval(arr,max=1.0, eps=1e-8):
    """ Scales all values in the numpy array to be between 0 and max """
    
    print 'scaling to interval [0,%s]'%max
    orig_shape = arr.shape
    flatt = arr.ravel() #flatten and then scale
    flatt -= flatt.min()
    flatt *= max/ (flatt.max()+eps)
    return flatt.reshape(orig_shape)
    
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


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)

    render_file = cfg.input.render_data #NB includes labels or sometimes not?
    render_file_has_labels = cfg.input.render_data_has_labels
    render_file_has_multi_labels = cfg.input.render_data_has_multi_labels
    weights_file = cfg.train.weights_file
    n_ins = cfg.shape.input_vector_length
    skip_trace_images = cfg.train.skip_trace_images
    
    input_vector_length = cfg.shape.input_vector_length
    mid_layer_sizes = list(cfg.shape.mid_layer_sizes)
    inner_code_length = cfg.shape.inner_code_length
    
    labels_file = cfg.output.labels_file
    codes_file = cfg.output.codes_file
    
    cost_method = cfg.train.cost;
    
    #load data to generate codes for
    if (render_file_has_multi_labels):
        raise RuntimeError('Peter broke this')
        dataset_x, dataset_labels = load_data_with_multi_labels(info['training_prefix']+'_0.pkl.gz')
    elif (render_file_has_labels):
        dataset_x, dataset_labels = load_data_with_labels(render_file)
    else:
        dataset_x = load_data_without_labels(render_file)
    
    # write labels
    # FIXME this should probably be done by the prepare.py step, no?
    #if (render_file_has_labels):
    #    write_csv_labels(dataset_labels, labels_file)
    
    # load weights file and initialize smh
    smh = load_model(cost_method, n_ins=input_vector_length,  mid_layer_sizes = mid_layer_sizes,
                    inner_code_length = inner_code_length, weights_file=weights_file)
    
    # check it is setup right by reconstructing the input
    #output_trace(smh, dataset_x, skip_trace_images)
    
    # run the input data forward through the smh
    codes = get_output_codes(smh, dataset_x)
    
    #output codes
    write_csv_codes(codes, codes_file)
    
    
    