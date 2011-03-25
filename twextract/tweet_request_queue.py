import os
import sys
import getopt
# move the working dir up one level so we can import hashmapd stuff
sys.path[0] = sys.path[0]+os.sep+'..'

import couchdb
import datetime

from hashmapd import LoadConfig,DefaultConfig

import inspect

#==============================================================================
# Manages the Tweet Request database.
#
# Contains methods to add request(s), or to get the next request off the db
# "Queue".
#==============================================================================
class TweetRequestQueue(object):
    
    n_pages = 3;
    
    def __init__(self,server_url='http://127.0.0.1:5984',db_name='hashmapd'):
        self.db = couchdb.Server(server_url)[db_name]
    
    # dequeue the front item (request) in the queue 
    def next(self):
        # 1) produce view of underway requests
        queue_view = self.db['_design/queue']['views']['underway_download_requests']['map'];
        results = self.db.query(queue_view)
        
        # 2) if the oldest request has been underway for too long (2m for now), return that (as it has probably failed)
        for result in results:
            row = self.db[result.id]
            try:
                if (datetime.datetime.strptime(row['started_time'],"%Y-%m-%dT%H:%M:%S.%f") > datetime.datetime.now()-datetime.timedelta(minutes=2)):
                    break
                row['started_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
                self.db[result.id] = row
                return row
            except KeyError:
                break
        
        # produce view of requests
        queue_view = self.db['_design/queue']['views']['queued_download_requests']['map'];
        results = self.db.query(queue_view)    
        
        # get the first result (first request in the queue), update the start time field, and begin work
        for result in results:
            row = self.db[result.id]
            row['started_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            self.db[result.id] = row
            return row
    
    # enqueue an item (request) to the back of the queue 
    def add(self,screen_name,page):
        self.db.save({'username':screen_name,'page':page,\
          'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),'type':'download_request'})
    
    # enqueue a series of requests (several pages) for a user to the back of the queue 
    def add_screen_name(self,screen_name):
        for page in xrange(1,self.n_pages+1):
            self.add(screen_name,page)

#==============================================================================
# Main method to add a tweet request (or a series of tweet requests) to the 
# db queue
#==============================================================================

class NoScreenNameError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def usage():
    return 'usage: tweet_request_queue                                          \n'+\
           '   screen_name        specifies screen_name to add to retrieve queue\n'+\
           '   [-p page]          specifies specific page of tweets to load     \n'+\
           '   [-c config]        specifies config file to load                 '

if __name__ == '__main__':
    # parse command line arguments
    try:
        if len(sys.argv) < 2:
            raise NoScreenNameError('must provide screen_name argument')
        
        opts,args = getopt.getopt(sys.argv[2:], "p:c:", ["page=","config="])
    except (NoScreenNameError,getopt.GetoptError), err:
        print >> sys.stderr, 'error: ' + str(err)
        print >> sys.stderr, usage()
        sys.exit(1)
    
    screen_name = sys.argv[1]
    page = None
    cfg = None
    for o,a in opts:
        if o in ("-c", "--config"):
            cfg = LoadConfig(a)
        elif o in ("-p", "--page"):
            page = int(a)
        else:
            assert False, "unhandled option"
    
    if cfg is None:
        cfg = DefaultConfig()
    
    # add request for user's tweet to the db queue
    tweet_request_queue = TweetRequestQueue(cfg.raw.couch_server_url,cfg.raw.request_queue_db)
    if page is None:
        tweet_request_queue.add_screen_name(screen_name)
    else:
        tweet_request_queue.add(screen_name,page)



