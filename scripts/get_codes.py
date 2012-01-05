import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import csv
from hashmapd.load_config import LoadConfig, DefaultConfig
from hashmapd.SMH import SMH
from hashmapd.utils import load_data_from_file

    
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

def get_output_codes(smh, data_x):
    print 'running input data forward through smh..'
    output_codes = smh.output_codes_given_x(data_x)
    return output_codes; #a 2d array consisting of 'smh' representation of each input row as a row of floats
    
def write_csv_labels(labels, output_file="out/labels.csv"):
    print 'writing labels to csv', output_file
    
    csv_writer = csv.writer(open(output_file, 'wb'), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for label in labels:
        csv_writer.writerow([str(label),]) 

def write_csv_codes(codes, output_file = "out/codes.csv"):
    print 'writing output codes to %s'%output_file
    
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

    #load data (and labels) to generate codes for
    dataset_x, dataset_labels = load_data_from_file(cfg.input.render_data)
 
    # write labels
    labels_file = cfg.output.labels_file
    write_csv_labels(dataset_labels, labels_file)
    
    # load weights file and initialize smh
    weights_file = cfg.train.weights_file
    n_ins = cfg.shape.input_vector_length
    input_vector_length = cfg.shape.input_vector_length
    mid_layer_sizes = list(cfg.shape.mid_layer_sizes)
    inner_code_length = cfg.shape.inner_code_length
    cost_method = cfg.train.cost_method;

    smh = load_model(cost_method, n_ins=input_vector_length,  mid_layer_sizes = mid_layer_sizes,
                    inner_code_length = inner_code_length, weights_file=weights_file)
     
    # run the input data forward through the smh
    codes = get_output_codes(smh, dataset_x)

    render_x = dataset_x[0:20]
    smh.output_trace_info(render_x,'render_data',False)

    # write output codes
    codes_file = cfg.output.codes_file
    write_csv_codes(codes, codes_file)
    
    
    