import os
import sys
import getopt
# move the working dir up one level so we can import hashmapd stuff
sys.path[0] = sys.path[0]+os.sep+'..'

import couchdb

from hashmapd import LoadConfig,DefaultConfig

#==============================================================================
# Manages the Tweet Request database.
#
# Contains methods to add request(s), or to get the next request off the db
# "Queue".
#==============================================================================
class TweetRequestQueue(object):
    
    n_pages = 3;
    
    def __init__(self,server_url='http://127.0.0.1:5984',db_name='tweet_request_queue'):
        self.db = couchdb.Server(server_url)[db_name]
    
    def next(self):
        front = self.db['front']
        back = self.db['back']
        
        front_val = front['value']
        
        # check to see if the db (queue) is empty
        if back['value'] < front_val:
            return None
        
        # note we can get away with this not being an atomic update as long
        # as this method never runs multiple times simultaneously
        next = self.db[str(front_val)]
        del self.db[str(front_val)]
        front['value'] += 1
        self.db['front'] = front
        return next
    
    def add(self,screen_name,page):
        back = self.db['back']
        
        # note we can get away with this not being an atomic update as long
        # as this method never runs multiple times simultaneously
        self.db[str(back['value']+1)] = {'screen_name':screen_name,'page':page}
        back['value'] += 1
        self.db['back'] = back
    
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



