
import sys
import getopt
import os
import csv
import cPickle
import gzip
import theano
import time, PIL.Image
import couchdb
import json

from hashmapd import *
from hashmapd.csv_unicode_helpers import UnicodeWriter

from couchdb.mapping import Document, LongField, DateField, FloatField, TextField, IntegerField, BooleanField
from couchdb import Server


class TextDoc(Document):
    screen_name = TextField()
    doc_type = TextField()
    text = TextField()

def upload_raw(cfg):
    # setup couchdb
    couch = Server(cfg.couchdb.server_url)
    db = couch[cfg.couchdb.database]
  
    #read json file
    dict = json.load(file(cfg.raw.raw_data_file))
    
    #copy data into couchdb format (in memory buffer)       
    buff = []
    for doctxt in dict:
        doc = TextDoc(
              screen_name = doctxt['screen_name'],
              text = doctxt['text'],
              doc_type = 'raw')
        buff.append(doc)
    
    #upload whole buffer as a single http POST
    #for greater efficiency on large data do this every 1000 or so rows
    db.update(buff)
    
def write_user_counts(cfg):
    
    couch = Server(cfg.couchdb.server_url)
    db = couch[cfg.couchdb.database]
    
    #get counts from view        
    view = db.view('couchapp/count', group=True)
    user_word_vector = []
    words = {}
    users = {}
    word_count = 0
    user_count = 0
    
    csv_word_counts = csv.writer(open(cfg.input.csv_data, 'wb'))
    csv_word_counts.writerow(('user_id', 'word_id','count'))

    for row in view:
        word = row.key[1]
        user = row.key[0]
        count = row.value
                
        word_id = words.get(word, word_count)
        if (word_id==word_count):
            word_count = word_count+1 #used default in above, so must be a new word
        words.setdefault(word,word_id)
        
        user_id = users.get(user, user_count)
        if (user_id==user_count):
            user_count = user_count+1 #used default in above, so must be a new user
        users.setdefault(user,user_id)
        
        print user_id, word_id, count, "::", user, word, count, user_count, word_count
        
        csv_word_counts.writerow((user_id, word_id,count))

    csv_words = UnicodeWriter(open(cfg.input.words_file, 'wb'))
    csv_words.writerow(('word_id', 'word'))
    for row in sorted(words.items(), key=lambda x: x[1]): #uses a great magic lambda that i do not fully understand MKT - courtesy http://stackoverflow.com/questions/613183/python-sort-a-dictionary-by-value
        csv_words.writerow(("%i"%row[1],row[0]))
        
    csv_users = csv.writer(open(cfg.input.users_file, 'wb'))
    csv_users.writerow(('user_id','screen'))
    for row in sorted(users.items(), key=lambda x: x[1]): 
        csv_users.writerow((row[1],row[0]))
    
    csv_user_labels = csv.writer(open(cfg.output.labels_file, 'wb'))
    for row in sorted(users.items(), key=lambda x: x[1]): 
        csv_user_labels.writerow(([row[0]]))
    
       
    print "User Count (put this into config, input.number_of_examples field) :",user_count
    print "Word Count (put this into config, shape.input_vector_length field):",word_count

def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])

    cfg = DefaultConfig() if (len(args)==0) else LoadConfig(args[0])

    if (args[1]=='upload'):
        upload_raw(cfg)
    if (args[1]=='write_counts'):
        write_user_counts(cfg)
 
if __name__ == '__main__':
    sys.exit(main())    
