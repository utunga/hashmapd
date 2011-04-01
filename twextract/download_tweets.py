import os
import sys
import getopt
# move the working dir up one level so we can import hashmapd stuff
if (not sys.path[0].endswith(os.sep+'..')):
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
from twextract.store_user import StoreUser
from twextract.store_tweets import StoreTweets
from twextract.request_queue import RequestQueue
import twextract.lib.tweepy as tweepy


min_hits = 50;
count = 100;
store_tweets = StoreTweets();
request_queue = RequestQueue();
api = tweepy.API(parser=tweepy.parsers.JSONParser());


#==============================================================================
# Factory to produce RetrieveTweet worker threads
#==============================================================================
class RetrieveTweetsFactory(object):
    
    def __init__(self,use_mock_data=False,save_data=False):
        self.use_mock_data = use_mock_data;
        self.save_data = save_data;
     
    def get_worker_thread(self,manager,screen_name,page,db,request_id):
        thread = RetrieveTweets(manager,screen_name,page,db,request_id)
        
        if (self.use_mock_data):
            thread.retrieve_data_from_twitter = Mock()
            thread.retrieve_data_from_twitter.side_effect = self.retrieve_pickled_data
        elif (self.save_data):
            thread.retrieve_data_from_twitter = Mock()
            thread.retrieve_data_from_twitter.side_effect = self.pickle_data
        
        return thread
    
    def retrieve_pickled_data(self,screen_name,page):
        f = open(sys.path[0]+os.sep+'twextract'+os.sep+'stored_tweets'+os.sep+str(screen_name)+str(page),'rb')
        tweet_data = cPickle.load(f)
        f.close()
        return tweet_data
    
    def pickle_data(self,screen_name,page):
        tweet_data = api.user_timeline(screen_name=screen_name,page=page,count=count,include_entities=True,trim_user=True,include_rts=False)
        f = open(sys.path[0]+os.sep+'twextract'+os.sep+'stored_tweets'+os.sep+str(screen_name)+str(page),'wb')
        cPickle.dump(tweet_data,f,cPickle.HIGHEST_PROTOCOL)
        f.close()
        return tweet_data


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
            self.manager.notify_failed(self,err)
            return
        
        # notify manager that data has been retrieved
        self.manager.notify_completed(self)
    
    def retrieve_data_from_twitter(self,screen_name,page):
        # TODO: we do not actually get back "count" tweets. we get back all the
        #       "non-retweets" over the last 100 tweets that the user has made.
        #       is this a big problem? (if so, there is a separate call to get
        #       retweets, but there doesn't appear to be a single call that gets
        #       both "statuses" and "retweets")
        return api.user_timeline(screen_name=screen_name,page=page,count=count,include_entities=True,trim_user=True,include_rts=False)

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
    
    def notify_completed(self,thread):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # report to the queue that the job finished successfully
        row = self.db[thread.request_id];
        request_queue.completed_request(row,thread.request_id);
        # create hash request if completed downloading user requests
        self.create_hash_request_if_finished(thread);
        # print notification of completion
        print 'Retrieved tweets ('+str(thread.screen_name)+','+str(thread.page)+')'
    
    def notify_failed(self,thread,err):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # report to the queue that the job failed
        row = self.db[thread.request_id]
        request_queue.failed_request(row,thread.request_id);
        # create hash request if completed downloading user requests
        self.create_hash_request_if_finished(thread);
        # print error message
        print >> sys.stderr, 'Error retrieving tweets ('+str(thread.screen_name)+','+str(thread.page)+'):\n'+str(err)
    
    def create_hash_request_if_finished(self,thread):
        # if there are no more pending download requests for this user,
        # create a new hash request for the user
        
        # TODO:
        #   - only want to create a hash if this was a user request 
        #   - may want to optimize the decision here a bit more.
        # (eg: has the user had a hash calculated recently?,
        #      how much more data has been added for this user since last hash?,
        #      etc.)
        results = self.db.view('queue/queued_user_download_requests', reduce=False)
        
        if len(results[thread.screen_name]) == 0:
            request_queue.add_hash_request(thread.screen_name);
            pass
    
    def run(self):
        # - obtain a twitter screen name from db that needs data downloaded
        # spawn a thread for each page of downloads
        while self.terminate == False:
            # get the next request
            next_request = request_queue.next('download')
            # if the queue is empty, check for new data (every 2 seconds for now)
            # (might want to just terminate here)
            if next_request == None:
                time.sleep(2)
                continue;
            
            screen_name = next_request['username']
            page = next_request['page']
            request_id = next_request.id
            
            # if there is no entry in the db for this user, create one 
            if screen_name not in self.db:
                thread = StoreUser(screen_name,self.db,api)
                thread.start()
            
            # check that there are enough twitter hits left
            # TODO: this may be able to be done more efficiently by examining 
            #       the header of the most recently returned data
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
    factory = RetrieveTweetsFactory();
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



