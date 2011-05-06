
import os, sys, time
import couchdb

server = couchdb.Server('http://localhost:5984')

# delete old db and create new one
if 'fake_txt' in server:
    server.delete('fake_txt')
    time.sleep(2.5);
db = server.create('fake_txt')

# use couchapp to push the views
time.sleep(1);
os.chdir(os.path.join(sys.path[0], 'queue'))
os.system('couchapp init')
time.sleep(1);
os.system('couchapp push http://localhost:5984/hashmapd')
time.sleep(1);
os.chdir(os.path.join(sys.path[0], 'tweets'))
os.system('couchapp init')
time.sleep(1);
os.system('couchapp push http://localhost:5984/hashmapd')
