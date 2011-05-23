import os
import sys
from optparse import OptionParser
from Queue import Queue
import threading
import time
import cPickle
import gzip
import datetime
import logging

import couchdb
import tweepy

from hashmapd.load_config import LoadConfig, DefaultConfig, BASEPATH
from twextract.store_user import StoreUser
from twextract.store_tweets import StoreTweets
from twextract.request_queue import RequestQueue


min_hits = 5
count = 100
store_tweets = StoreTweets()
logger = logging.getLogger('download_tweets')
logger.setLevel(logging.DEBUG)
logged_empty_queue = False

#Default limit
rate_limit = {'hourly_limit': 350, 'remaining_hits': 350, 'reset_time_in_seconds': int(time.time()) + 360}

#Rate semaphore
rate_semaphore = threading.BoundedSemaphore()

MIN_BACKOFF_TIME = 0.5
MAX_BACKOFF_TIME = 600
backoff_time = 0.5
SMALL_PAUSE = 0.1
wait_until = time.time()

class RetrieveTweets(threading.Thread):
    """
    Get the specified page of tweets for the specified user, and store them in
    the specified db
   """ 
    def __init__(self, manager, screen_name, page, db, request_id):
        threading.Thread.__init__(self)
        self.manager = manager
        self.screen_name = screen_name
        self.page = page
        self.db = db
        self.request_id = request_id
        logger.debug('RetrieveTweets: Downloading tweets of %s, page %s, from database %s'%(self.screen_name, self.page, self.db))
    
    def run(self):
        try:
            # obtain tweet data
            tweet_data = self.retrieve_data_from_twitter(self.screen_name, self.page)
            # store the tweets in specified db
            store_tweets.store(self.screen_name, tweet_data, self.db)
        except Exception, err:
            # notify manager of error
            self.manager.notify_failed(self, err)
            return
        
        # notify manager that data has been retrieved
        self.manager.notify_completed(self)
    
    def retrieve_data_from_twitter(self, screen_name, page):
        # TODO: we do not actually get back "count" tweets. we get back all the
        #       "non-retweets" over the last 100 tweets that the user has made.
        #       is this a big problem? (if so, there is a separate call to get
        #       retweets, but there doesn't appear to be a single call that gets
        #       both "statuses" and "retweets")
        global wait_until, backoff_time
        try:
            return api.user_timeline(screen_name=screen_name, page=page, count=count, include_entities=True, trim_user=True, include_rts=False)
            backoff_time = MIN_BACKOFF_TIME
            wait_until = time.time() + backoff_time
        except Exception, err:
            # TODO: depending on the err type, should handle this in different ways
            #       (for now just raise the error)
            logger.exception(str(err))
            backoff_time = min(backoff_time*2, MAX_BACKOFF_TIME)
            if backoff_time == MAX_BACKOFF_TIME:
                logger.error('Maximum backoff time (%s) reached'%MAX_BACKOFF_TIME)
            wait_until = time.time() + backoff_time
            raise

class Manager(threading.Thread):
    """
    Loop until user terminates program. Obtain tweet requests from the queue and
    spawn worker threads to retrieve the data.
 
    Blocks when the maximum number of simultanous requests are underway.
    Currently busy-waits when there are no requests on the queue.
    Also busy-waits when the twitter rate limit is reached.
    """
    
    def __init__(self, server_url='http://127.0.0.1:5984', db_name='hashmapd',\
            max_simultaneous_requests=5):
        threading.Thread.__init__(self)
        self.db = couchdb.Server(server_url)[db_name]
        self.request_queue = RequestQueue(server_url=server_url, db_name=db_name)
        self.worker_threads = []
        self.semaphore = threading.Semaphore(max_simultaneous_requests)
        self.terminate = False
        logger.debug('Manager: downloading tweets from %s, database %s'%(server_url, db_name))
    
    def notify_completed(self,thread):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # report to the queue that the job finished successfully
        row = self.db[thread.request_id]
        self.request_queue.completed_request(row, thread.request_id)
        # create hash request if completed downloading user requests
        self.create_hash_request_if_finished(thread)
        # print notification of completion
        logger.info('Retrieved tweets (' + str(thread.screen_name)+',' + str(thread.page) + ')')
    
    def notify_failed(self,thread,err):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # report to the queue that the job failed
        row = self.db[thread.request_id]
        self.request_queue.failed_request(row, thread.request_id)
        # create hash request if completed downloading user requests
        self.create_hash_request_if_finished(thread)
        # print error message
        logger.error('Error retrieving tweets (' + str(thread.screen_name) + ',' + str(thread.page) + '):\n' + str(err))
    
    def create_hash_request_if_finished(self, thread):
        # if there are no more pending download requests for this user,
        # create a new hash request for the user
        
        # TODO:
        #   - do we only want to create a hash request if this was a user request?
        #   - also, we may want to optimize the decision here a bit more
        # (eg: has the user had a hash calculated recently?,
        #      how much more data has been added for this user since last hash?,
        #      etc.)
        results = self.db.view('queue/queued_user_download_requests', reduce=False)
        
        if len(results[thread.screen_name]) == 0:
            self.request_queue.add_hash_request(thread.screen_name)
    
    def run(self):
        # obtain a twitter screen name from db that needs data downloaded
        # spawn a thread for each page of downloads
        global logged_empty_queue, wait_until
        while self.terminate == False:
            # get the next request
            logger.debug('Manager: get the next request')
            next_request = self.request_queue.next('download')
            try:
                logger.debug('Manager: Next request is for %s, page %s'%(next_request['username'], next_request['page']))
            except:
                pass
            # if the queue is empty, check for new data (every 1 second for now)
            # (might want to just terminate here)
            
            if next_request == None:
                if not logged_empty_queue:
                    logger.info('Request queue is empty - no more users to download')
                    logged_empty_queue = True
                time.sleep(1)
                continue 
            logged_empty_queue = False
            screen_name = next_request['username']
            page = next_request['page']
            request_id = next_request.id
            
            # if there is no entry in the db for this user, create one 
            if screen_name not in self.db:
                thread = StoreUser(screen_name, self.db, api)
                thread.start()
            
            # check that there are enough twitter hits left
            # TODO: this may be able to be done more efficiently by examining 
            #       the header of the most recently returned data, or keeping
            #       a running total
            logger.debug('Acquire rate semaphore for %s, page %s'%(screen_name, page))
            rate_semaphore.acquire()
            try:
                rate_limit = api.rate_limit_status()
                logger.debug('%s hits left'%(rate_limit['remaining_hits']))
                if rate_limit['remaining_hits'] == 0:
                    wait_until = max(rate_limit['reset_time_in_seconds'], wait_until)
                    logger.info('No more hits, wait for %s seconds'%(wait_until - time.time()))
                if wait_until > int(time.time()):
                    logger.debug('Sleep for %s seconds'%(wait_until - int(time.time())))
                    time.sleep(wait_until - int(time.time()))
            
                # acquire a lock before creating new thread
                self.semaphore.acquire()
                # spawn a worker thread to retreive the specified tweet data
                time.sleep(SMALL_PAUSE)
                thread = RetrieveTweets(self, screen_name, page, self.db, request_id)
                self.worker_threads.append(thread)
                logger.debug('Start thread downloading tweets for %s, page %s'%(screen_name, page))
                thread.start()
            finally:
                rate_semaphore.release()
        
        # wait until all threads have finished 
        while len(self.worker_threads) > 0:
            thread = self.worker_threads[0]
            thread.join()
        
        # determine no hits left after completion
        rate_limit = api.rate_limit_status()
        logger.info('Exited download_tweets.py, hits left: ' + str(rate_limit['remaining_hits']) + ' (reset time: ' + str(rate_limit['reset_time']+')'))
    
    def exit(self):
        self.terminate = True

if __name__ == '__main__':
    """
    Main method to intialize and run the Manager indefinitely
    """
    parser = OptionParser()
    parser.add_option("-c", "--cfg", help="Config file name", default=os.path.join(BASEPATH, "base"))
    parser.add_option("-u", "--url", help="Couchdb url", default=None)
    parser.add_option("-d", "--database",  help="Couchdb database", default=None)
    parser.add_option("-s", "--secrets", help = "Oauth secrets", default=os.path.join(BASEPATH, "secrets"))
    parser.add_option("-l", "--log", help="Log file", default='download_tweets.log')

    options, args = parser.parse_args()
    cfg = LoadConfig(options.cfg)
    if options.url is not None:
        cfg.raw.couch_server_url = options.url
    if options.database is not None:
        cfg.raw.couch_db = options.database
        
    # authenticate with oauth
    secrets_cfg = LoadConfig(options.secrets)
    auth = tweepy.OAuthHandler(secrets_cfg.auth.consumer_token, secrets_cfg.auth.consumer_secret)
    auth.set_access_token(secrets_cfg.auth.session_key, secrets_cfg.auth.session_secret)
    
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
    
    # determine no hits left before starting
    rate_limit = api.rate_limit_status()
    log_file = logging.FileHandler(options.log)
    log_stream = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    log_file.setFormatter(log_formatter)
    log_stream.setFormatter(log_formatter)
    logger.addHandler(log_file)
    logger.addHandler(log_stream)
    logger.info('Starting download_tweets.py, hits left: '+str(rate_limit['remaining_hits']))
     
    # run the manager
    manager = Manager(cfg.raw.couch_server_url, cfg.raw.couch_db, cfg.raw.max_simultaneous_requests)
    manager.start()
    
    # keep running until user types 'exit', then terminate nicely
    # (ie: allow all currently running jobs to complete before closing) 
    while True:
        input = raw_input("type \'exit\' to terminate:\n")
        if input == 'exit':
            manager.exit()
            logger.info('Terminated')
            break


