
import os, sys, time
import couchdb

SERVER = 'http://localhost:5984'
DATABASE = 'hashmapd'

server = couchdb.Server(SERVER)

# Create database if it doesn't already exist
if not DATABASE in server:
    db = server.create(DATABASE)

# use couchapp to push the views
for view in ('queue', 'tweets', 'tokens'):
    time.sleep(1);
    os.chdir(os.path.join(sys.path[0], view))
    os.system('couchapp init')
    time.sleep(1);
    os.system('couchapp push %s/%s'%(SERVER, DATABASE))

