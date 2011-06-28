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
from hashmapd.tsne import TSNE

def write_csv_coords(coords, output_file="out/coords.csv"):
    #coords = scale_to_interval(coords, max=100)
    
    print 'writing coordinates to csv'
    csv_writer = csv.writer(open(output_file, 'wb'), delimiter=',')
    for r in xrange(len(coords)):
        csv_writer.writerow(coords[r].astype('|S12')) # format with 10dp accuracy (but no '-e' format stuff)

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)

    codes_file = cfg.output.codes_file
    coords_file = cfg.output.coords_file
    
    codes = numpy.genfromtxt(codes_file, dtype=numpy.float32, delimiter=',')
    codes = codes[:,1:]
        
    tsne = TSNE(perplexity=cfg.tsne.perplexity, desired_dims=cfg.tsne.desired_dims)
    tsne.initialize_with_codes(codes)
    tsne.fit(iterations=cfg.tsne.initial_fit_iterations)
    tsne.save_coords_to_file(coords_file)
    
    #tsne.load_from_file(coords_file,codes_file)
    #tsne.fit(iterations=2)
    
    #test_code = [0.1350030452,0.4128168225,0.0014129921,0.7547346354,0.0068102819,0.6216894388,0.9996289015,0.8628810048,0.0004052414,0.0012938380,0.9998107553,0.0000006208,0.2459984124,0.0001938931,0.0103854276,0.0001564398,0.0000000090,0.9995579720,0.9649902582,0.0000025402,0.9946812987,0.9264854193,0.9999329448,0.0095445570,0.0054685692,0.9955748916,0.9433483481,0.0002042586,0.0430774689,0.7664549351]
    #tsne.get_coord_for_code(test_code, iterations =2)
    
