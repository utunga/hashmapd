import os, sys, time
import couchdb
import json

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
from couchdb.mapping import Document, TextField

class TextDoc(Document):
    screen_name = TextField()
    doc_type = TextField()
    text = TextField()

def upload_raw(cfg):
    # setup couchdb
    couch = couchdb.Server(cfg.raw.couch_server_url)
    db = couch[cfg.raw.couch_db]
  
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
    print 'uploaded data from %s to couchdb ' % cfg.raw.raw_data_file

def reinit_db(cfg):
    # setup couchdb
    server = couchdb.Server(cfg.raw.couch_server_url)
    db_name = cfg.raw.couch_db

    # delete old db and create new one
    if db_name in server:
        server.delete(db_name)
        time.sleep(.1);
    db = server.create(db_name)

    print 'recreated database %s/%s' % (cfg.raw.couch_server_url, db_name)
    
    # use couchapp to push the views
    time.sleep(.1);
    os.chdir(os.path.join(sys.path[0], 'couch', 'fake_txt'))
    os.system('couchapp init')
    time.sleep(.1);
    os.system('couchapp push %s/%s' % (cfg.raw.couch_server_url, db_name))
    os.chdir(sys.path[0])

    print 'pushed views using couchapp'
    
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()
    cfg = LoadConfig(options.config)

    reinit_db(cfg)
    upload_raw(cfg)
    
