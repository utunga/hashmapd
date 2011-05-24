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
from hashmapd.csv_unicode_helpers import UnicodeReade

class WordDoc(Document):
      token = TextField()
      screen_name = TextField()
      count = LongField()
      x_coord = FloatField()
      y_coord = FloatField()
      included =BooleanField()
      word_usercount = LongField()
      user_wordcount = LongField()
      overall_count = LongField()
      word_prob = FloatField()
      doc_type = TextField()
      #"text","screen_name","count","x_coord","y_coord","included","word_user_count","overall_count","user_word_count","word_prob"

      def calc_word_prob(self):
            self.word_prob = self.count*1.00 / self.user_wordcount
            self.doc_type = "token_n_coords"

def read_word_data_into_couch(cfg):
      """
      Reads in data from csv input and pushes it into couchdb
      """
      print "attempting to read " + cfg.raw.words_with_coords
      
      couch = Server(cfg.couchdb.server_url)
      db = couch[cfg.couchdb.database]
      
      buff = []
      unicodeReader = UnicodeReader(open(cfg.raw.words_with_coords,'r'))
      for i,row in enumerate(unicodeReader):
          if (i>0): #skip header row
              doc = WordDoc(
                  token = row[0], #row["text"],
                  screen_name = row[1], #row["screen_name"],
                  count = row[2], #row["count"],
                  x_coord = row[3], #row["x_coord"],
                  y_coord = row[4], #row["y_coord"],
                  included = row[5], #row["included"],
                  word_usercount = row[6], #row["word_user_count"],
                  overall_count = row[7], #row["overall_count"],
                  user_wordcount = row[8], #row["user_word_count"],
                  )
              #print doc.word_prob
              if (doc.overall_count<30000):
                  doc.calc_word_prob()
                  buff.append(doc)
                  if i % 2000==0:  
                      db.update(buff)
                      buff = []
                      print 'copied up to row '+ str(i) + '..';
        
      db.update(buff)
      print 'copied up to row '+ str(i) + '..';
            

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)

    
    #read in csv and post data to couchdb
    read_word_data_into_couch(cfg);
    

if __name__ == '__main__':
    sys.exit(main())    