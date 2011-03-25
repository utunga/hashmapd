import sys
import os
# move the working dir up one level so we can import hashmapd stuff
sys.path[0] = sys.path[0]+os.sep+'..'

from hashmapd import LoadConfig,DefaultConfig

import couchdb

#===============================================================================
# Accesses the default config's request queue database, and sets the initial
# front and back fields (if they don't already exist), so that the queueing
# functions can be used.
#===============================================================================

if __name__ == '__main__':
    cfg = DefaultConfig()
    db = couchdb.Server(cfg.raw.couch_server_url)[cfg.raw.request_queue_db];
    
    try:
        db['front']
    except couchdb.http.ResourceNotFound:
        db['front'] = {'value':0};
    
    try:
        db['back']
    except couchdb.http.ResourceNotFound:
        db['back'] = {'value':-1};
