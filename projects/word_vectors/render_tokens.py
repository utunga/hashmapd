
import sys
import getopt
import os
import csv
import cPickle
import gzip
#import theano
import time, PIL.Image
import couchdb
import numpy
import re
from pylab import plot, draw, figure, imshow, xlabel, ylabel, cm, show, axis, savefig, text, clf
from scipy import stats, mgrid, c_, reshape, random, rot90

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
from hashmapd.querycouch import QueryCouch
from hashmapd.csv_unicode_helpers import UnicodeWriter


def clean_filename(filename):
    """util function"""
    return re.sub("[^a-zA-Z]", "", filename)

def output_top_word_per_square(query):
    
    csv_writer_coords = csv.writer(open('out/couch_coords.csv', 'wb'))
    csv_writer_labels = UnicodeWriter(open('out/couch_labels.csv', 'wb'))

    for x_coord in range(-200,200):
        for y_coord in range(-200,200):
            row = query.top_token_for_square(x_coord, y_coord)
            print row
            if (len(row)>2):
                csv_writer_coords.writerow([x_coord,y_coord])
                csv_writer_labels.writerow([row[2]])

def output_all_tokens(query):
    
    rows = query.all_tokens()
    for row in rows:
        print row
        
def output_top_square_per_token(query):
    
    csv_writer_coords = csv.writer(open('out/couch_coords.csv', 'wb'))
    csv_writer_labels = UnicodeWriter(open('out/couch_labels.csv', 'wb'))
    
    rows = query.all_tokens()
    for row in rows:
        (x_coord,y_coord,token) = query.top_square_for_token(row[1][0])
        print (x_coord,y_coord,token)
        csv_writer_coords.writerow([x_coord,y_coord])
        csv_writer_labels.writerow([token])
          
def render_token(query, token='Yoga', output_file='out/yoga_density.png'):
    locations = numpy.array(query.locations_for_token(token))
        
    m1 = locations[:,0] # x-coords
    m2 = locations[:,1] # y-coords
    
    # Perform a kernel density estimator on the coords in data.
  
    # FIXME: temporary hard code the max/min so all plots are on same scale
    xmin = -150
    xmax = 150
    ymin = -150
    ymax = 150
    #xmin = m1.min()
    #xmax = m1.max()
    #ymin = m2.min()
    #ymax = m2.max()
    
    
    X, Y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = c_[X.ravel(), Y.ravel()]
    values = c_[m1, m2]
    kernel = stats.kde.gaussian_kde(values.T)
    Z = reshape(kernel(positions.T).T, X.T.shape)
    clf() #kinda insane that one has to do this *Before* you render but hrmm.. anyway necessary in case this function gets caleld twice
    imshow(rot90(Z), cmap=cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    
    # Plot the locations (assumes each 'mention' has same weight - which is a shame)
    plot(m1, m2, 'k.', markersize=1)
    text(xmin,ymax-15,"density map for '" + token + "'")
     
    axis('equal')
    print 'saving density map for ', token, 'to ', output_file
    savefig(output_file)    

def render_top_tokens(query, skip=0, topN = 100):
    rows = query.all_tokens(topN)
    for row in rows[skip:topN]:
        token = row[1][0]
        render_token(cfg, token, 'density_maps/'+clean_filename(token.lower())+'.png')
        
def main(argv = sys.argv):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    #cfg = LoadConfig(options.config)
    #validate_config(cfg)
    
    couchdb_server_url = "http://127.0.0.1:5984"
    couchdb = "hashmapd_bak"
    query = QueryCouch(couchdb_server_url, couchdb)
    
    render_token(query, 'Yoga', 'out/yoga_density.png')
    #render_token(cfg, 'Lol', 'out/lol_density.png')
    #render_top_tokens(cfg, 0, 1000)
    #output_all_tokens(cfg)

    
    #rows = query.non_english_screennames()
    #print "<html><body>"
    #for row in rows:
    #    print "<a href='http://twitter.com/" + row[1] +"' >" , row[0] , "." , row[1] , "</a><br />"
    #print "</body></html>"

    #print "count,screen_name"
    #for row in rows:
    #   print row[0],",",row[1]
    
if __name__ == '__main__':
    sys.exit(main())    