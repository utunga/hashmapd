import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import csv
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

    # old/standard method - fixed perplexity
    #tsne = TSNE(perplexity=cfg.tsne.perplexity, desired_dims=cfg.tsne.desired_dims)
    #tsne.initialize_with_codes(codes)
    #tsne.fit(iterations=cfg.tsne.initial_fit_iterations)
    #tsne.save_coords_to_file(coords_file)

    # reducing perplexity method

    target_perplex = cfg.tsne.perplexity
    desired_dims=cfg.tsne.desired_dims
    target_iterations = cfg.tsne.initial_fit_iterations

    # FIXME2 slightly braindead way to work out epochs expected (just run through it)
    total_epochs = 0
    if (target_perplex>=(len(codes)/2)):
        total_epochs=1
    print len(codes)
    perplexity = len(codes)/2
    while perplexity > target_perplex:
        perplexity = perplexity / 2
        total_epochs = total_epochs+1

    iterations_per = target_iterations / total_epochs
    print "will run for %i total epochs at %i iterations each " %(total_epochs, iterations_per)

    # initialize
    tsne = TSNE(desired_dims=desired_dims)
    tsne.initialize_with_codes(codes)

    #fit with reducing perplexity, ending up at target complexity
    perplexity = len(codes)/2
    while perplexity > target_perplex:
        print 'training with perplexity', perplexity
        tsne.perplexity = perplexity
        tsne.fit(iterations_per)
        perplexity = perplexity / 2

    print 'done training'
    tsne.save_coords_to_file(coords_file)
