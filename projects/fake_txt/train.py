import os, sys, getopt

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
from hashmapd.SMH import train_SMH


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
    
    

    
        
    
    
