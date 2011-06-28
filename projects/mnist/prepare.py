import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image, math
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

from hashmapd.utils import tiled_array_image
from hashmapd.load_config import LoadConfig

def load_and_truncate_mnist(raw_dataset_file, data_folder):
 
    print '...  loading full data from '+ raw_dataset_file
    f = gzip.open(raw_dataset_file,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    for (name, data, count) in [
            ('training', train_set, cfg.input.number_for_training),
            ('validation', valid_set, cfg.input.number_for_validation),
            ('testing', test_set, cfg.input.number_for_testing),
            ('render', train_set, cfg.input.number_for_training)]:
        (x, y) = data
        filename = os.path.join('data', name+'_data_0.pkl.gz')
        f = gzip.open(filename, 'wb')
        print '...  pickling and zipping {0} data to {1}'.format(
                name, filename)
        data = (x[:count], None, y[:count])
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL) # no sums/labels


#def test_truncated_mnist(data_folder):
#    f = gzip.open(os.path.join(data_folder,TRAINING_FILE+'_0.pkl.gz'),'rb')
#    train_set,sums,labels = cPickle.load(f)
#    f.close()
#    
#    # Plot filters after each training epoch
#    
#    # Construct image from the weight matrix 
#    image = tiled_array_image( train_set)
#    image.save('trace/truncated_input.png')
    
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)
    
    load_and_truncate_mnist('raw/mnist.pkl.gz', 'data')
