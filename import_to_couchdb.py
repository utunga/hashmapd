
import sys
import getopt
import os
import csv
import cPickle
import gzip
import theano
import time, PIL.Image
import couchdb

from hashmapd import *
from hashmapd.csv_unicode_helpers import UnicodeReader

from struct import *
from numpy import *
from couchdb.mapping import Document, LongField, DateField, FloatField, TextField, IntegerField, BooleanField
from couchdb import Server

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
              doc.calc_word_prob()
              #print doc.word_prob
              buff.append(doc)
              if i % 2000==0:  
                  db.update(buff)
                  buff = []
                  print 'copied up to row '+ str(i) + '..';
        
      db.update(buff)
      print 'copied up to row '+ str(i) + '..';
            

def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])
    #validate_config(cfg)
    
    #read in csv and post data to couchdb
    read_word_data_into_couch(cfg);
    

if __name__ == '__main__':
    sys.exit(main())    