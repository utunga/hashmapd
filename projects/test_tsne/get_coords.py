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

from hashmapd.tsne import TSNE
from hashmapd.render import Render

def write_csv_coords(coords, output_file="out/coords.csv"):
    #coords = scale_to_interval(coords, max=100)
    
    print 'writing coordinates to csv'
    csv_writer = csv.writer(open(output_file, 'wb'), delimiter=',')
    for r in xrange(len(coords)):
        csv_writer.writerow(coords[r].astype('|S12')) # format with 10dp accuracy (but no '-e' format stuff)

if __name__ == '__main__':

    codes_file = 'test/test_codes.csv'
    coords_file = 'out/test_coords.csv'
    
    codes = numpy.genfromtxt(codes_file, dtype=numpy.float32, delimiter=',')
    codes = codes[:,1:]
    
    for perplexity in xrange(20):
        try:
            tsne = TSNE(perplexity=perplexity, desired_dims=2)
            tsne.initialize_with_codes(codes)
            tsne.fit(iterations=500)
            tsne.save_coords_to_file(coords_file)
            
            density_plot_file = 'out/test_map_%i.png'%perplexity
            labels_file = 'test/test_labels.csv'
        
            render = Render(coords_file, labels_file)
            render.plot_density(density_plot_file)
        except:
            print 'failed to compute with perplexity %i'%perplexity
            
