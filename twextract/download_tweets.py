import os
import sys
import getopt
# move the working dir up one level so we can import hashmapd stuff
sys.path[0] = sys.path[0]+os.sep+'..'

from mock import Mock

from Queue import Queue
import threading
import time

import cPickle
import gzip

import datetime

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
# Factory to produce RetrieveTweet worker threads
#==============================================================================
class RetrieveTweetsFactory(object):
    
    def __init__(self,use_mock_data=False,save_mock_data=False):
        self.use_mock_data = use_mock_data;
        self.save_mock_data = save_mock_data;
     
    def get_worker_thread(self,manager,screen_name,page,db,request_id):
        thread = RetrieveTweets(manager,screen_name,page,db,request_id)
        
        if (self.use_mock_data):
            thread.retrieve_data_from_twitter = Mock()
            thread.retrieve_data_from_twitter.side_effect = self.retrieve_pickled_data
        elif (self.save_mock_data):
            thread.retrieve_data_from_twitter = Mock()
            thread.retrieve_data_from_twitter.side_effect = self.pickle_data
        
        return thread
    
    def retrieve_pickled_data(thread,screen_name,page):
        f = open('twextract'+os.sep+'stored_tweets'+os.sep+str(screen_name)+str(page),'rb')
        tweet_data = cPickle.load(f)
        f.close()
        return tweet_data
    
    def pickle_data(thread,screen_name,page):
        tweet_data = thread.retrieve_data_from_twitter(self.screen_name,self.page);
        f = open('twextract'+os.sep+'stored_tweets'+os.sep+str(screen_name)+str(page),'wb')
        cPickle.dump(tweet_data,f,cPickle.HIGHEST_PROTOCOL)
        f.close()


#==============================================================================
# Get the specified page of tweets for the specified user, and store them in
# the specified db
#==============================================================================
class RetrieveTweets(threading.Thread):
    
    def __init__(self,manager,screen_name,page,db,request_id):
        threading.Thread.__init__(self)
        self.manager = manager
        self.screen_name = screen_name
        self.page = page
        self.db = db        
        self.request_id = request_id
    
    def run(self):
        try:
            # obtain tweet data
            tweet_data = self.retrieve_data_from_twitter(self.screen_name,self.page);
            # store the tweets in specified db
            store_tweets.store(self.screen_name,tweet_data,self.db)
        except Exception, err:
            # notify manager of error
            self.manager.notifyFailed(self,err)
        # notify manager that data has been retrieved
        self.manager.notifyCompleted(self)
    
    def retrieve_data_from_twitter(self,screen_name,page):
        return api.user_timeline(screen_name=screen_name,count=count,page=page,include_entities=True,trim_user=True,include_rts=False)

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
    
    def __init__(self,server_url='http://127.0.0.1:5984',db_name='hashmapd',\
            retrieval_factory=RetrieveTweetsFactory(),max_simultaneous_requests=5):
        threading.Thread.__init__(self)
        self.db = couchdb.Server(server_url)[db_name]
        self.retrieval_factory = retrieval_factory
        self.worker_threads = []
        self.semaphore = semaphore = threading.Semaphore(max_simultaneous_requests)
        self.terminate = False
    
    def notifyCompleted(self,thread):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # write finished time
        row = self.db[thread.request_id]
        row['completed_time'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.db[thread.request_id] = row
        # print notification of completion
        print 'Retrieved tweets ('+str(thread.screen_name)+','+str(thread.page)+')'
    
    def notifyFailed(self,thread,err):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # clear started time, so that job will be restarted
        # TODO: (is this a good idea? what if the job keeps failing over and over for some reason?)  
        row = self.db[thread.request_id]
        del row['started_time']
        self.db[thread.request_id] = row
        # print error message
        print >> sys.stderr, 'Error retrieving tweets ('+str(thread.screen_name)+','+str(thread.page)+'):\n'+str(err)
    
    def run(self):
        # - obtain a twitter screen name from file
        # spawn a thread for each page
        while self.terminate == False:
            # get the next request
            next_request = tweet_request_queue.next()
            # if the queue is empty, check for new data (every 5 seconds for now)
            # (might want to just terminate here)
            if next_request == None:
                time.sleep(5)
                continue;
            
            screen_name = next_request['username']
            page = next_request['page']
            request_id = next_request['_id']
            
            # check that there are enough twitter hits left
            # TODO: this may be able to be done more efficiently by examining 
            #         the header of the most recently returned data
            rate_limit = tweepy.api.rate_limit_status()
            if rate_limit['remaining_hits'] < min_hits:
                print >> sys.stderr, 'only '+str(rate_limit['remaining_hits'])+' hits allowed until '+str(rate_limit['reset_time'])
                break
            
            # acquire a lock before creating new thread
            self.semaphore.acquire()
            # spawn a worker thread to retreive the specified tweet data
            thread = self.retrieval_factory.get_worker_thread(self,screen_name,page,self.db,request_id)
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
    
    
    def exit(self):
        self.terminate = True;


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
        opts,args = getopt.getopt(sys.argv[1:], "c:", ["config="])
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
        else:
            assert False, "unhandled option"
    
    if cfg is None:
        cfg = DefaultConfig()
    
    # run the manager
    factory = RetrieveTweetsFactory(use_mock_data=True);
    thread = Manager(cfg.raw.couch_server_url,cfg.raw.couch_db,factory,cfg.raw.max_simultaneous_requests);
    thread.start()
    
    # keep running until user types 'exit', then terminate nicely
    # (ie: allow all currently running jobs to complete before closing) 
    while True:
        input = raw_input("type \'exit\' to terminate:\n");
        if input == 'exit':
            thread.exit();
            print 'terminating ...'
            break



