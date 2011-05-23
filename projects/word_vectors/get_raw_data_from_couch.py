import os, sys, getopt
import numpy, time, cPickle, gzip, PIL.Image
import csv
import couchdb
import json

from couchdb.mapping import Document, LongField, DateField, FloatField, TextField, IntegerField, BooleanField
from couchdb import Server

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
from hashmapd.csv_unicode_helpers import UnicodeWriter
    
def write_user_counts(cfg):
    
    couch = Server(cfg.raw.couch_server_url)
    db = couch[cfg.raw.couch_db]
    
    #get counts from view        
    view = db.view('fake_txt/count', group=True)
    user_word_vector = []
    words = {}
    users = {}
    word_count = 0
    user_count = 0
    
    csv_word_counts = csv.writer(open(cfg.raw.csv_data, 'wb'))
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

    csv_words = UnicodeWriter(open(cfg.raw.words_file, 'wb'))
    csv_words.writerow(('word_id', 'word'))
    for row in sorted(words.items(), key=lambda x: x[1]): #uses a great magic lambda that i do not fully understand MKT - courtesy http://stackoverflow.com/questions/613183/python-sort-a-dictionary-by-value
        csv_words.writerow(("%i"%row[1],row[0]))
        
    csv_users = csv.writer(open(cfg.raw.users_file, 'wb'))
    csv_users.writerow(('user_id','screen'))
    for row in sorted(users.items(), key=lambda x: x[1]): 
        csv_users.writerow((row[1],row[0]))
    
    csv_user_labels = csv.writer(open(cfg.output.labels_file, 'wb'))
    for row in sorted(users.items(), key=lambda x: x[1]): 
        csv_user_labels.writerow(([row[0]]))
    
    print "User Count (put this into config.. input.number_of_examples field) :",user_count
    print "Word Count (put this into config.. shape.input_vector_length field):",word_count

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)

    write_user_counts(cfg)
 