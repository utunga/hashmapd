import os
import sys
import getopt
# move the working dir up one level so we can import hashmapd stuff
sys.path[0] = sys.path[0]+'\\..'

from Queue import Queue
import threading
import time

import cPickle
import gzip

import couchdb

from hashmapd import LoadConfig,DefaultConfig
from twextract.store_tweets import StoreTweets
from twextract.tweet_request_queue import TweetRequestQueue
import twextract.lib.tweepy as tweepy


min_hits = 50;
count = 10; # TODO: why is this only returning half as many as is requested?
store_tweets = StoreTweets();
tweet_request_queue = TweetRequestQueue();
api = tweepy.API(parser=tweepy.parsers.JSONParser());


#==============================================================================
# Get the specified page of tweets for the specified user, and store them in
# the specified db
#==============================================================================
class RetrieveTweets(threading.Thread):
    
    def __init__(self,manager,screen_name,page,db,use_test_data=False,save_test_data=False):
        threading.Thread.__init__(self)
        self.manager = manager
        self.screen_name = screen_name
        self.page = page
        self.db = db        
        self.use_test_data = use_test_data
        self.save_test_data = save_test_data
    
    def run(self):
        try:
            # obtain tweet data
            if self.use_test_data:
                tweet_data = self.retrieve_pickled_data();
            else:
                tweet_data = self.retrieve_data_from_twitter();
                if self.save_test_data: self.pickle_data(tweet_data);
            
            # store the tweets in specified db
            store_tweets.store(self.screen_name,tweet_data,self.db)
            
            # notify manager that data has been retrieved
            self.manager.notifyCompleted(self)
        
        except Exception, err:
            # notify manager of error
            self.manager.notifyFailed(self,self.screen_name,self.page,err)
    
    def retrieve_data_from_twitter(self):
        return api.user_timeline(screen_name=self.screen_name,count=count,page=self.page,include_entities=True,trim_user=True,include_rts=False)
    
    def pickle_data(self,tweet_data):
        f = open('twextract'+os.sep+'stored_tweets'+os.sep+str(self.screen_name)+str(self.page),'wb')
        cPickle.dump(tweet_data,f,cPickle.HIGHEST_PROTOCOL)
        f.close()
    
    def retrieve_pickled_data(self):
        f = open('twextract'+os.sep+'stored_tweets'+os.sep+str(self.screen_name)+str(self.page),'rb')
        tweet_data = cPickle.load(f)
        f.close()
        return tweet_data


#==============================================================================
# Loop until user terminates program. Obtain tweet requests from the queue and
# spawn worker threads to retrieve the data.
# 
# Blocks when the maximum number of simultanous requests are underway.
# Currently busy-waits when there are no requests on the queue.
# 
# Exits with error message if twitter rate limit exceeded.
#==============================================================================
class Manager(threading.Thread):
    
    def __init__(self,server_url='http://127.0.0.1:5984',db_name='fake_txt',\
                 max_simultaneous_requests=5,use_test_data=False,save_test_data=False):
        threading.Thread.__init__(self)
        self.db = couchdb.Server(server_url)[db_name]
        self.worker_threads = []
        self.semaphore = semaphore = threading.Semaphore(max_simultaneous_requests)
        self.use_test_data = use_test_data
        self.save_test_data = save_test_data
    
    def notifyCompleted(self,thread):
        self.worker_threads.remove(thread)
        self.semaphore.release()
    
    def notifyFailed(self,thread,screen_name,page,err):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # TODO: better error handling here
        print >> sys.stderr, 'Error retrieving tweets ('+str(screen_name)+','+str(page)+'):\n'+str(err)
    
    def run(self):
        # - obtain a twitter screen name from file
        # spawn a thread for each page
        while terminate == False:
            # get the next request
            next_request = tweet_request_queue.next()
            # if the queue is empty, check for new data (every 5 seconds for now)
            # (might want to just terminate here)
            if next_request == None:
                time.sleep(5)
                continue;
            
            screen_name = next_request['screen_name']
            page = next_request['page']
            
            # check that there are enough twitter hits left
            # TODO: this may be able to be done more efficiently by examining 
            #         the header of the most recently returned data
            rate_limit = tweepy.api.rate_limit_status()
            if rate_limit['remaining_hits'] < min_hits:
                print >> sys.stderr, 'only '+rate_limit['remaining_hits']+' hits allowed until '+rate_limit['reset_time']
                break
            
            # acquire a lock before creating new thread
            self.semaphore.acquire()
            # spawn a worker thread to retreive the specified tweet data
            thread = RetrieveTweets(self,screen_name,page,self.db,self.use_test_data,self.save_test_data)
            self.worker_threads.append(thread)
            thread.start()
        
        # wait until all threads have finished 
        while len(self.worker_threads) > 0:
            thread = self.worker_threads[0]
            thread.join()
        
        # determine no hits left after completion
        limit = tweepy.api.rate_limit_status()
        print ''
        print 'exited download_tweets.py, hits left: '+str(limit['remaining_hits'])
        print '(reset time: '+str(limit['reset_time']+')')


#==============================================================================
# Main method to intialize and run the Manager indefinitely
#==============================================================================

def usage():
    return 'usage: get_tweets                                                   \n'+\
           '   [-c config]        specifies config file to load                 '

if __name__ == '__main__':
    # determine no hits left before starting
    limit = tweepy.api.rate_limit_status()
    print 'starting download_tweets.py, hits left: '+str(limit['remaining_hits'])
    print ''
    
    # parse command line arguments
    try:
        opts,args = getopt.getopt(sys.argv[1:], "c:ts", ["config=","test","save"])
    except getopt.GetoptError, err:
        print >> sys.stderr, 'error: ' + str(err)
        print >> sys.stderr, usage()
        sys.exit(1)
    
    use_test_data = False
    save_test_data = False
    cfg = None
    for o,a in opts:
        if o in ("-c", "--config"):
            cfg = LoadConfig(a)
        elif o in ("-t", "--test"):
            use_test_data = True
        elif o in ("-s", "--save"):
            save_test_data = True
        else:
            assert False, "unhandled option"
    
    if cfg is None:
        cfg = DefaultConfig()
    
    # run the manager
    terminate = False
    thread = Manager(cfg.raw.couch_server_url,cfg.raw.couch_db,cfg.raw.max_simultaneous_requests,use_test_data,save_test_data);
    thread.start()
    
    # keep running until user types 'exit', then terminate nicely
    # (ie: allow all currently running threads to complete before closing) 
    while True:
        input = raw_input("type \'exit\' to terminate:\n");
        if input == 'exit':
            terminate = True
            print 'terminating ...'
            break



