import os
import sys
import getopt
import couchdb
from uuid import uuid4
import datetime
import inspect

from hashmapd.load_config import LoadConfig, DefaultConfig

class RequestQueue(object):
    """
    Manages the Tweet Request database.

    Contains methods to add request(s), or to get the next request off the db
    "Queue".
    """
    
    n_pages = 3
    
    def __init__(self, server_url='http://127.0.0.1:5984',db_name='hashmapd'):
        self.db = couchdb.Server(server_url)[db_name]
    
    # dequeue the front item (request) in the queue (queue_name = 'download' or 'hash')
    def next(self,queue_name):
        # 1) produce view of underway requests
        results = self.db.view('queue/underway_'+queue_name+'_requests', reduce=False, descending=False)
        
        # 2) if the oldest request has been underway for too long (30s for now - may want to reduce this),
        #    return that (as it has probably failed)
        for result in results:
            row = self.db[result.id]
            try:
                if (datetime.datetime.strptime(row['started_time'],"%Y-%m-%dT%H:%M:%S.%f") > datetime.datetime.now()-datetime.timedelta(seconds=30)):
                    break
                self.started_request(row,result.id)
                return row
            except KeyError:
                break
        
        # produce view of requests
        results = self.db.view('queue/queued_'+queue_name+'_requests', reduce=False, descending=False)
        
        # get the first result (first request in the queue), update the start time field, and begin work
        for result in results:
            row = self.db[result.id]
            self.started_request(row,result.id)
            return row
    
    # enqueue an item (hash request) to the back of the queue 
    def add_hash_request(self,username,priority=0):
        # note that we create a new hash request regardless of whether or not 
        # another one already exists (completed or otherwise)
        self.db[uuid4().hex] = {'username':username,'priority':priority,\
          'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),'doc_type':'hash_request'}
    
    # lower priority requests will be processed first (can be negative)
    
    # enqueue an item (download request) to the back of the queue 
    def add_download_request(self,username,page,priority=0):
        # note that we create a new download request regardless of whether or not 
        # another one already exists (completed or otherwise)  
        self.db[uuid4().hex] = {'username':username,'page':page,'priority':priority,\
          'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),'doc_type':'download_request'}
    
    # enqueue a series of download requests (several pages) for a user to the back of the queue 
    def add_download_requests_for_username(self,username,priority=0):
        for page in xrange(1,self.n_pages+1):
            self.add_download_request(username,page,priority)
    
    # mark a request as started (work is underway) 
    def started_request(self,row,request_id):
        row['started_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.db[request_id] = row
    
    # mark a request as completed
    def completed_request(self,row,request_id):
        row['completed_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.db[request_id] = row
        # TODO: store/update a value indicating the most recently downloaded tweet here
        #       (which could then be used to determine what new info needs to be downloaded)
    
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
    


class NoScreenNameError(Exception):
    """
    Main method to add a tweet request (or a series of tweet requests) to the 
    db queue
    """
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
    request_queue = RequestQueue(cfg.raw.couch_server_url,cfg.raw.request_queue_db)
    if page is None:
        request_queue.add_download_requests_for_username(screen_name)
    else:
        request_queue.add_download_request(screen_name,page)



