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

from hashmapd.utils import tile_raster_images
from hashmapd.load_config import LoadConfig, DefaultConfig, dict_to_cfg

TRAINING_FILE = "training_data"
VALIDATION_FILE = "validation_data"
TESTING_FILE = "testing_data"

def load_and_truncate_mnist(batch_size, raw_dataset_file=None, data_folder=None, render_file=None):
 
    print '...  loading full data from '+ raw_dataset_file
    f = gzip.open(raw_dataset_file,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    train_set_x = train_set[0]
    valid_set_x = valid_set[0]
    test_set_x = test_set[0]

    # supervised data aka labels (not used in training)
    train_set_y = train_set[1]
    valid_set_y = valid_set[1]
    test_set_y = test_set[1]
    
    #set up config file

    
    train_cutoff = cfg.input.number_for_training
    validate_cutoff = cfg.input.number_for_validation #train_cutoff+cfg.input.number_for_validation
    test_cutoff = cfg.input.number_for_testing #validate_cutoff+cfg.input.number_for_testing
    batch_size = cfg.train.train_batch_size
    
    print '...  truncating to smaller set' #truncate the data so it fits in the damn gpu
    train_set_x = test_set_x[0:train_cutoff]
    valid_set_x = valid_set_x[0:validate_cutoff]
    test_set_x = train_set_x[0:test_cutoff]
    
    mean_doc_size = train_set_x.sum(axis=1).mean()

    data_info = {'training_prefix': os.path.join(data_folder, TRAINING_FILE),
        'n_training_files': 1,
        'n_training_batches':train_cutoff/batch_size,
        'validation_prefix':  os.path.join(data_folder, VALIDATION_FILE),
        'n_validation_files': 1,
        'n_validation_batches': validate_cutoff/batch_size,
        'testing_prefix':  os.path.join(data_folder, TESTING_FILE),
        'n_testing_files': 1,
        'n_testing_batches': test_cutoff/batch_size,
        'batches_per_file': (train_cutoff+validate_cutoff+test_cutoff)/batch_size,
        'mean_doc_size': mean_doc_size,
    }
    
    #dict_to_cfg(data_info, 'info', 'data.cfg')

        
    #print '...  pickling and zipping truncated, unsupervised data to '+ data_folder
    #
    #f = gzip.open(os.path.join(data_folder,TRAINING_FILE+'_0.pkl.gz'),'wb')
    #cPickle.dump((train_set_x,train_set_x.sum(axis=1),[]),f, cPickle.HIGHEST_PROTOCOL) # no sums/labels
    #f.close()
    #
    #f = gzip.open(os.path.join(data_folder,VALIDATION_FILE+'_0.pkl.gz'),'wb')
    #cPickle.dump((valid_set_x,valid_set_x.sum(axis=1),[]),f, cPickle.HIGHEST_PROTOCOL) # no sums/labels
    #f.close()
    #
    #f = gzip.open(os.path.join(data_folder,TESTING_FILE+'_0.pkl.gz'),'wb')
    #cPickle.dump((test_set_x,test_set_x.sum(axis=1),[]),f, cPickle.HIGHEST_PROTOCOL) # no sums/labels
    #f.close()
    
    print '...  pickling and zipping truncated, render data (with labels) to '+ render_file

    f = gzip.open(render_file,'wb')
    cPickle.dump((train_set_x,train_set_y),f, cPickle.HIGHEST_PROTOCOL) # no sums/labels
    f.close()


def test_truncated_mnist(data_folder):
    f = gzip.open(os.path.join(data_folder,TRAINING_FILE+'_0.pkl.gz'),'rb')
    train_set,sums,labels = cPickle.load(f)
    f.close()
    
    # Plot filters after each training epoch
    
    # Construct image from the weight matrix 
    image = PIL.Image.fromarray(tile_raster_images( train_set,
             img_shape = (28,28),tile_shape = (30,30), 
             tile_spacing=(1,1)))
    image.save('trace/truncated_input.png')
    
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)
    
    batch_size = cfg.train.train_batch_size
    raw_dataset_file = cfg.raw.supervised_data_file
    data_folder = cfg.input.data_folder
    render_file = cfg.input.render_file
    load_and_truncate_mnist(batch_size, raw_dataset_file, data_folder, render_file)
    test_truncated_mnist(data_folder)
