import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import csv
import couchdb
from couchdb.mapping import Document, LongField, DateField, FloatField, TextField, IntegerField, BooleanField
from couchdb import Server

#from struct import *
#from numpy import *

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
from hashmapd.csv_unicode_helpers import UnicodeReader

class UserDoc(Document):
      screen_name = TextField()
      x_coord = FloatField()
      y_coord = FloatField()
      trained_as_id = LongField()
    
def read_user_coords_into_couch(cfg):
    """
    Reads in data from csv input and pushes it into couchdb
    """
    print "attempting to read " + cfg.raw.user_coords_file
    
    couch = Server(cfg.couchdb.couch_server_url)
    db = couch[cfg.couchdb.couch_db]
    
    print 'will upload to '+ cfg.couchdb.couch_server_url + '/' + cfg.couchdb.couch_db;
     
    buffer = []
    unicodeReader = UnicodeReader(open(cfg.raw.user_coords_file,'r'))
    for i,row in enumerate(unicodeReader):
        if (i>0): #skip header row
          doc = UserDoc(
              trained_as_id = row[0], #row["user_id"],
              screen_name = row[1], #row["screen_name"],
              x_coord = row[2], #row["x_coord"],
              y_coord = row[3] #row["y_coord"],
              )
          doc.id = "twuser_" + row[1] # important to give docs id so updates are idempotent
          doc["doc_type"] = "twuser"
          buffer.append(doc)
          if i % 1000==0:  
              db.update(buffer)
              buffer = []
              print 'uploaded user data, to row '+ str(i) + '..';
      
    db.update(buffer) #
    print 'uploaded user data, to row '+ str(i) + '..';
            

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    
    cfg = LoadConfig(options.config)
    read_user_coords_into_couch(cfg);
