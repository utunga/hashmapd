import os, sys, getopt
import numpy, cPickle, gzip
import theano

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

from hashmapd.load_config import LoadConfig
from hashmapd.SMH import SMH


def load_data_from_file(dataset_file):
    ''' Loads the dataset

    :type dataset_file: string
    :param dataset_file: the path to the dataset
    '''
    print >>sys.stderr, '... loading data from file ' + dataset_file
    
    if dataset_file.endswith('.npy'):
        #x = numpy.lib.format.open_memmap(dataset_file, mode='r') # or mode='c'  
        x = numpy.lib.format.read_array(open(dataset_file, 'rb'))
        assert x.dtype is numpy.dtype(theano.config.floatX), (x.dtype, theano.config.floatX)
        y = None
    else:
        f = gzip.open(dataset_file, 'rb')
        (x, x_sums, y) = cPickle.load(f)
        x = numpy.asarray(x, dtype=theano.config.floatX)
    return (x, y)


def load_data(datadir, part):
    """Try any known file formats in which the array might be stored"""
    file_prefix = os.path.join(datadir, part + '_data')
    for file_suffix in ['.npy', '_0.pkl.gz']:
        filename = file_prefix + file_suffix
        if os.path.exists(filename):
            return load_data_from_file(filename)
    raise RuntimeError('No {0} data found in {1}/'.format(part, datadir))
            

def load_training_arrays(datadir, input_vector_length=None):
    """Load the arrays from the data directory
    
    :param datadir: path to the directory holding the data files
    :param input_vector_length: number of columns expected in each file
    
    Returns [train, valid, test]"""
            
    result = []
    for part in ['training', 'validation', 'testing']:
        (x, y) = load_data(datadir, part)
        if input_vector_length is None:
            input_vector_length = x.shape[1]
        elif x.shape[1] != input_vector_length:
            raise ValueError('Expected {0} columns of {1} data but found {2}'.format(
                    input_vector_length, part, x.shape[1]))
        result.append(x)
    return result


def train_SMH(datadir, mid_layer_sizes, inner_code_length, first_layer_type, **kw):
    """Create a SMH and train it with the data in 'datadir'"""
        
    for (alternate_name, suggest) in [
            ('skip_trace_during_training', 'skip_trace_images'),
            ('cost', 'cost_method'),
            ('train_batch_size', 'batch_size'),
            ('n_ins', 'input_vector_length'),]:
        if alternate_name in kw:
            value = kw.pop(alternate_name)
            if suggest in kw:
                print >>sys.stderr, "Config setting {0}={1} was ignored, but {2}={3}".format(
                        alternate_name, value, suggest, kw[suggest])
            else:
                kw[suggest] = value
    
    data = load_training_arrays(datadir, kw.pop('input_vector_length'))
    data = [(a, a.sum(axis=1)[:, numpy.newaxis]) for a in data]
    (training_data, validation_data, test_data) = data
    (x, x_sums) = training_data
    
    smh = SMH(
            numpy_rng = numpy.random.RandomState(123),
            mean_doc_size = x.sum(axis=1).mean(), 
            first_layer_type = first_layer_type, 
            n_ins = x.shape[1],
            mid_layer_sizes = mid_layer_sizes,
            inner_code_length = inner_code_length,
            )
            
    smh.train(training_data, validation_data, training_data, **kw)

    return smh


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)
    smh = train_SMH('data',
            mid_layer_sizes = list(cfg.shape.mid_layer_sizes), 
            inner_code_length = cfg.shape.inner_code_length, 
            **cfg.train)


    #double check that save/load worked OK

    #def load_model(**kw):
    #    numpy_rng = numpy.random.RandomState(212)
    #    smh = SMH(numpy_rng = numpy_rng, **kw)
    #    smh.unroll_layers(cost_method, 0); #need to unroll before loading model otherwise doesn't work
    #    save_file=open(weights_file, 'rb')
    #    smh_params = cPickle.load(save_file)
    #    save_file.close()
    #    smh.load_model(smh_params)
    #    return smh


    #info = LoadConfig('data')['info']
    #testing_data = load_data(info['testing_prefix']+'_0.pkl.gz')
    
    #smh.save_model(weights_file=weights_file)
    #smh.output_trace_info(testing_data[0][:3], 'test_weights_b4_restore', skip_trace_images)
    
    #smh2 = load_model(cost_method = cost_method, first_layer_type = first_layer_type, n_ins=n_ins,  mid_layer_sizes = mid_layer_sizes,
    #                inner_code_length = inner_code_length, weights_file=weights_file)
    #output_trace_info(smh2, testing_data[0][:3], 'test_weights_restore', skip_trace_images)
    
    

    
        
    
    
