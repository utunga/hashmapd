import os
import sys
import getopt
# move the working dir up one level so we can import hashmapd stuff
if (not sys.path[0].endswith(os.sep+'..')):
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
    
    # dequeue the front item (request) in the queue (queue_name = 'download' or 'hash')
    def next(self,queue_name):
        # 1) produce view of underway requests
        queue_view = self.db['_design/queue']['views']['underway_'+queue_name+'_requests']['map'];
        results = self.db.query(queue_view)
        
        # 2) if the oldest request has been underway for too long (30s for now - may want to reduce this),
        #    return that (as it has probably failed)
        for result in results:
            row = self.db[result.id]
            try:
                if (datetime.datetime.strptime(row['started_time'],"%Y-%m-%dT%H:%M:%S.%f") > datetime.datetime.now()-datetime.timedelta(seconds=30)):
                    break
                self.started_request(row,result.id);
                return row
            except KeyError:
                break
        
        # produce view of requests
        queue_view = self.db['_design/queue']['views']['queued_'+queue_name+'_requests']['map'];
        results = self.db.query(queue_view)    
        
        # get the first result (first request in the queue), update the start time field, and begin work
        for result in results:
            row = self.db[result.id]
            self.started_request(row,result.id);
            return row
    
    # enqueue an item (hash request) to the back of the queue 
    def add_hash_request(self,username):
        self.db.save({'username':username,\
          'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),'doc_type':'hash_request'})
    
    # enqueue an item (download request) to the back of the queue 
    def add_download_request(self,username,page):
        self.db.save({'username':username,'page':page,\
          'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),'doc_type':'download_request'})
    
    # enqueue a series of download requests (several pages) for a user to the back of the queue 
    def add_download_requests_for_username(self,username):
        for page in xrange(1,self.n_pages+1):
            self.add_download_request(username,page)
    
    # mark a request as started (work is underway) 
    def started_request(self,row,request_id):
        row['started_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.db[request_id] = row
    
    # mark a request as completed
    def completed_request(self,row,request_id):
        row['completed_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.db[request_id] = row
    
    # work on a request has failed - so clear the started field, and increment
    # the number of attempts field
    def failed_request(self,row,request_id):
        # clear started field
        del row['started_time']
        # increment number of attempts field
        try:
            fails = row['attempts']
        except KeyError:
            fails = 0
        row['attempts'] = (fails+1)
        # store the update row in the db
        self.db[request_id] = row
    

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
        tweet_request_queue.add_download_requests_for_username(screen_name)
    else:
        tweet_request_queue.add_download_request(screen_name,page)



