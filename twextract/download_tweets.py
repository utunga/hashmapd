import logging
from optparse import OptionParser
import os
import threading
import time

import couchdb
import tweepy

from hashmapd.load_config import LoadConfig, BASEPATH
from twextract.store_user import StoreUser
from twextract.store_tweets import StoreTweets
from twextract.request_queue import RequestQueue


MIN_HITS = 3
COUNT = 100
store_tweets = StoreTweets()
logger = logging.getLogger('download_tweets')
logger.setLevel(logging.INFO)

#Semaphores
MAX_SIMULTANEOUS_REQUESTS = 5
lock = threading.Lock()
semaphore = threading.BoundedSemaphore(MAX_SIMULTANEOUS_REQUESTS)

class Status(object):
    MIN_BACKOFF_TIME = 0.5
    MAX_BACKOFF_TIME = 600
    SMALL_PAUSE = 0.1
    BIG_PAUSE = 10

    def __init__(self):
        self.rate_limit = {'remaining_hits': 0, 'reset_time_in_seconds': 360}
        self.terminate = False
        self.wait_until = time.time()
        self.backoff_time = 0.5

    def info(self, message, thread):
        logger.info(message + '(%s, %s, %s, %s)'%(thread.screen_name, 
            thread.page, self.rate_limit['remaining_hits'], thread.getName()))

    def error(self, message, thread):
        logger.info(message + '(%s, %s, %s, %s)'%(thread.screen_name, 
            thread.page, self.rate_limit['remaining_hits'], thread.getName()))

    def reset_wait_until(self):
        self.backoff_time = self.MIN_BACKOFF_TIME
        self.wait_until = time.time() + self.backoff_time

    def backoff(self):
        self.backoff_time = min(self.backoff_time*2, self.MAX_BACKOFF_TIME)
        logger.debug('Backoff for %i seconds (%s)'%(self.backoff_time, threading.current_thread().getName()))
        if self.backoff_time == self.MAX_BACKOFF_TIME:
            logger.info('Maximum backoff time (%s) reached'%self.MAX_BACKOFF_TIME)
        self.wait_until = time.time() + self.backoff_time

    def wait(self):
        if time.time() + 60 < self.wait_until:
            logger.debug('Waiting for %s seconds (%s)'%(int(self.wait_until - time.time()), threading.current_thread().getName()))
        while time.time() < self.wait_until:
            time.sleep(self.SMALL_PAUSE)
            if self.terminate:
                logger.debug('Someone wants to kill me - stop waiting around (%s)'%threading.current_thread().getName())
                break

    def get_rate_limit(self):
        while not self.terminate:
            try:
                self.rate_limit = api.rate_limit_status()
                logger.debug('get_rate_limit: number of hits - %s, time to wait - %s (%s)'%(self.rate_limit['remaining_hits'], 
                    self.rate_limit['reset_time_in_seconds'], threading.current_thread().getName()))
                break
            except tweepy.error.TweepError, err:
                logger.info('Rate limit request raised an error: %s'%err)
                for i in range(self.BIG_PAUSE):
                    time.sleep(1)
                    if self.terminate:
                        break

    def hits(self):
        self.wait()
        self.get_rate_limit()
        remaining_hits = self.rate_limit['remaining_hits']
        while remaining_hits < MIN_HITS and not self.terminate:
            reset_time = min(self.rate_limit['reset_time_in_seconds'] - time.time(), 3600)
            self.wait_until = max(reset_time + time.time(), self.wait_until)
            logger.info('No more hits, wait for %s seconds'%(self.wait_until - time.time()))
            self.wait()
            if self.terminate:
                return False
            self.get_rate_limit()
            remaining_hits = self.rate_limit['remaining_hits']
        logger.debug('%s hits left (%s)'%(self.rate_limit['remaining_hits'], threading.current_thread().getName()))
        if self.rate_limit['remaining_hits'] > 2*MAX_SIMULTANEOUS_REQUESTS:
            return True
        else:
            return False

status = Status()

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
        # acquire a semaphore before creating new thread
        semaphore.acquire()
        logger.debug('RetrieveTweets: Downloading tweets of %s, page %s, from database %s'%(self.screen_name, self.page, self.db))
    
    def run(self):
        try:
            # obtain tweet data. The logic here is to check the rate limit status
            # before each request for data. If there are no hits left, then wait a while
            # before trying again. If there are only a few requests left, then require a 
            # lock before hitting twitter - the downloading becomes single threaded when
            # the number of hits gets close to zero.
            hits = status.hits()
            if not status.terminate:
                if hits:
                    tweet_data = self.retrieve_data_from_twitter(self.screen_name, self.page)
                else:
                    with lock:
                        tweet_data = self.retrieve_data_from_twitter(self.screen_name, self.page)
    
                # store the tweets in specified db
                store_tweets.store(self.screen_name, tweet_data, self.db)
                # set next wait time
                status.reset_wait_until()
                # notify manager that data has been retrieved
                self.manager.notify_completed(self)
            semaphore.release()
        except Exception as err:
            semaphore.release()
            if err.__class__.__name__ == 'TweepError' and str(err) in ('Not found', 'Not authorized'):
                logger.info('Page not found (%s, %s)'%(self.screen_name, self.page))
                self.manager.delete_request_doc(self)
            elif err.__class__.__name__ == 'TweepError' and str(err).startswith('Rate limit exceeded'):
                status.backoff()
                self.manager.notify_failed(self, err, notify_error=True)
            elif err.__class__.__name__ == 'TweepError' and str(err) == 'Twitter error response: status code = 503':
                status.backoff()
                self.manager.notify_failed(self, err)
            else:
                logger.exception(str(err))
                self.manager.notify_failed(self, err, notify_error=True)
                raise 
    
    def retrieve_data_from_twitter(self, screen_name, page):
        #       We do not actually get back "count" tweets. we get back all the
        #       "non-retweets" over the last 100 tweets that the user has made.
        #       is this a big problem? (if so, there is a separate call to get
        #       retweets, but there doesn't appear to be a single call that gets
        #       both "statuses" and "retweets")
        return api.user_timeline(screen_name=screen_name, page=page, count=COUNT, include_entities=True, trim_user=True, include_rts=False)

class Manager(threading.Thread):
    """
    Loop until user terminates program. Obtain tweet requests from the queue and
    spawn worker threads to retrieve the data.
 
    Blocks when the maximum number of simultanous requests are underway.
    Currently busy-waits when there are no requests on the queue.
    Also busy-waits when the twitter rate limit is reached.
    """
    
    def __init__(self, server_url='http://127.0.0.1:5984', db_name='hashmapd'):
        threading.Thread.__init__(self)
        self.db = couchdb.Server(server_url)[db_name]
        self.request_queue = RequestQueue(server_url=server_url, db_name=db_name)
        logger.debug('Manager: downloading tweets from %s, database %s'%(server_url, db_name))
    
    def notify_completed(self, thread):
        # report to the queue that the job finished successfully
        row = self.db[thread.request_id]
        self.request_queue.completed_request(row, thread.request_id)
        # create hash request if completed downloading user requests
        self.create_hash_request_if_finished(thread)
        # print notification of completion
        status.info('Manager: retrieved tweets', thread)
    
    def notify_failed(self, thread, err, notify_error=False):
        #backoff
        status.backoff()
        # report to the queue that the job failed
        row = self.db[thread.request_id]
        self.request_queue.failed_request(row, thread.request_id)
        # create hash request if completed downloading user requests
        self.create_hash_request_if_finished(thread)
        # print error message
        if notify_error:
            status.error('Manager: error (%s) retrieving tweets'%err, thread)
        else:
            status.info('Manager: error (%s) retrieving tweets'%err, thread)
    
    def delete_request_doc(self, thread):
        doc = self.db[thread.request_id]
        self.db.delete(doc)
        logger.debug('Deleted request for missing tweets (' + str(thread.screen_name) + ',' + str(thread.page) + ')')

    def create_hash_request_if_finished(self, thread):
        # if there are no more pending download requests for this user,
        # create a new hash request for the user
        results = self.db.view('queue/queued_user_download_requests', reduce=False)
        if len(results[thread.screen_name]) == 0:
            self.request_queue.add_hash_request(thread.screen_name)
    
    def run(self):
        # obtain a twitter screen name from db that needs data downloaded
        # spawn a thread for each page of downloads
        while status.terminate == False:
            # get the next request
            logger.debug('Manager: get the next request')
            next_request = self.request_queue.next('download')
            if next_request == None:
                logger.info('Request queue is empty - no more users to download')
                status.terminate = True
                continue
            logger.debug('Manager: Next request is for %s, page %s'%(next_request['username'], next_request['page']))
            screen_name = next_request['username']
            page = next_request['page']
            request_id = next_request.id
            
            # if there is no entry in the db for this user, create one 
            if screen_name not in self.db:
                logger.debug('Create record for user %s'%screen_name)
                hits = status.hits()
                if hits:
                    thread = StoreUser(screen_name, self.db, api)
                else:
                    with lock:
                        thread = StoreUser(screen_name, self.db, api)
                thread.start()
            
            thread = RetrieveTweets(self, screen_name, page, self.db, request_id)
            logger.debug('Start thread %s downloading tweets for %s, page %s'%(screen_name, page, thread.getName()))
            thread.setDaemon(True)
            thread.start()
        
        # wait until all threads have finished
        main_thread = threading.current_thread()
        for thread in threading.enumerate():
            if thread is main_thread:
                continue
            logging.debug('Joining thread %s', thread.getName())
            thread.join() 
        
        logger.info('Exited download_tweets.py')
    
    def exit(self):
        status.terminate = True
        logger.info('Terminating threads')

if __name__ == '__main__':
    #Intialize and run the Manager indefinitely
    parser = OptionParser()
    parser.add_option("-c", "--cfg", help="Config file name", default=os.path.join(BASEPATH, "base"))
    parser.add_option("-u", "--url", help="Couchdb url", default=None)
    parser.add_option("-d", "--database",  help="Couchdb database", default=None)
    parser.add_option("-s", "--secrets", help = "Oauth secrets", default=os.path.join(BASEPATH, "secrets"))
    parser.add_option("-l", "--log", help="Log file", default='download_tweets.log')
    parser.add_option("-m", "--mode", help="Logging mode (debug, info, or error)", default='info')

    options, args = parser.parse_args()
    cfg = LoadConfig(options.cfg)
    if options.url is not None:
        cfg.raw.couch_server_url = options.url
    if options.database is not None:
        cfg.raw.couch_db = options.database
    mode_lookup = {'debug': logging.DEBUG, 'info':logging.INFO, 'error':logging.ERROR}
    options.mode = mode_lookup.get(options.mode, logging.DEBUG)
       
    # authenticate with oauth
    secrets_cfg = LoadConfig(options.secrets)
    auth = tweepy.OAuthHandler(secrets_cfg.auth.consumer_token, secrets_cfg.auth.consumer_secret)
    auth.set_access_token(secrets_cfg.auth.session_key, secrets_cfg.auth.session_secret)
    
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
    
    
    # setup logging
    logger.setLevel(options.mode)
    log_file = logging.FileHandler(options.log)
    log_stream = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    log_file.setFormatter(log_formatter)
    log_stream.setFormatter(log_formatter)
    logger.addHandler(log_file)
    logger.addHandler(log_stream)
    logger.info('Starting download_tweets.py')
     
    # run the manager
    manager = Manager(cfg.raw.couch_server_url, cfg.raw.couch_db) 
    manager.start()

    # keep running until user types 'exit', then terminate nicely
    # (ie: allow all currently running jobs to complete before closing) 
    while True:
        text = raw_input("type \'exit\' to terminate\n")
        if text == 'exit':
            manager.exit()
            logger.info('Terminated')
            break
        time.sleep(0.5)


