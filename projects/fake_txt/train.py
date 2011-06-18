import os, sys, getopt
import numpy, cPickle, gzip
import theano

import matplotlib.pyplot as plt

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
from hashmapd.utils import tiled_array_image
from hashmapd.SMH import SMH


def load_data(dataset_file):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    print '... loading data from file ' + dataset_file
    
    if dataset_file.endswith('.npy'):
        x = numpy.lib.format.open_memmap(dataset_file, mode='r')
        assert x.dtype is numpy.dtype(theano.config.floatX), (x.dtype, theano.config.floatX)
        x_sums = x.sum(axis=1)
    else:
        f = gzip.open(dataset_file, 'rb')
        (x, x_sums, y) = cPickle.load(f)
        x = numpy.asarray(x, dtype=theano.config.floatX)

    # build a replcated 2d array of sums so operations can be performed efficiently   # HUH?
    x_sums = numpy.asarray(numpy.array([x_sums]*(x.shape[1])).transpose(), dtype=theano.config.floatX)
    
    return (x, x_sums)


def load_model(**kw):
    numpy_rng = numpy.random.RandomState(212)
    smh = SMH(numpy_rng = numpy_rng, **kw)
    smh.unroll_layers(cost_method, 0); #need to unroll before loading model otherwise doesn't work
    save_file=open(weights_file, 'rb')
    smh_params = cPickle.load(save_file)
    save_file.close()
    smh.load_model(smh_params)
    return smh


def train_SMH(datadir="data", **kw):
    for part in ['training', 'validation', 'testing']:
        for suffix in ['.npy', '_0.pkl.gz']:
            filename = os.path.join(datadir, part + '_data' + suffix)
            print filename
            if os.path.exists(filename):
                kw[part+'_data'] =  load_data(filename)
                break
        else:
            raise RuntimeError('No {0} data found in {1}/'.format(part, datadir))
        
    kw['mean_doc_size'] = kw['training_data'][1].mean()
    
    init_kw = {}
    for arg in ['first_layer_type', 'mean_doc_size', 'inner_code_length', 'mid_layer_sizes', 'n_ins']:
        if arg in kw:
            init_kw[arg] = kw.pop(arg)
    init_kw['numpy_rng'] = numpy.random.RandomState(123)
    
    smh = SMH(**init_kw) 
    smh.train(**kw)

    return smh


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)
    smh = train_SMH(
            batch_size = cfg.train.train_batch_size, 
            pretraining_epochs = cfg.train.pretraining_epochs,
            training_epochs = cfg.train.training_epochs,
            mid_layer_sizes = list(cfg.shape.mid_layer_sizes),
            inner_code_length = cfg.shape.inner_code_length,
            n_ins = cfg.shape.input_vector_length,
            first_layer_type = cfg.train.first_layer_type,
            method = cfg.train.method,
            k = cfg.train.k,
            noise_std_dev = cfg.train.noise_std_dev,
            cost_method = cfg.train.cost,
            skip_trace_images = cfg.train.skip_trace_images,
            weights_file = cfg.train.weights_file)
    
    #double check that save/load worked OK
    #info = LoadConfig('data')['info']
    #testing_data = load_data(info['testing_prefix']+'_0.pkl.gz')
    
    #smh.save_model(weights_file=weights_file)
    #smh.output_trace_info(testing_data[0][:3], 'test_weights_b4_restore', skip_trace_images)
    
    #smh2 = load_model(cost_method = cost_method, first_layer_type = first_layer_type, n_ins=n_ins,  mid_layer_sizes = mid_layer_sizes,
    #                inner_code_length = inner_code_length, weights_file=weights_file)
    #output_trace_info(smh2, testing_data[0][:3], 'test_weights_restore', skip_trace_images)
    
    

    
        
    
    
