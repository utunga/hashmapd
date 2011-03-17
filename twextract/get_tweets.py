from Queue import Queue
import threading
import os

import cPickle
import gzip

import couchdb

import lib.tweepy as tweepy


# get the most recent 1,000 tweets of the specified user
class StoreTweets(threading.Thread):
    
    def __init__(self,api,screen_name,page,db):
        threading.Thread.__init__(self)        
        self.api = api
        self.screen_name = screen_name
        self.page = page
        self.db = db
        
    def run(self): # TODO: make count 100
        # statuses = self.api.user_timeline(screen_name=self.screen_name,count=count,page=self.page)
        
        # TODO: move this quick hack to save/load the users tweets into a separate method(s)
        #       (so we don't reach the tweet download limit)
        # f = open('twextract'+os.sep+'stored_tweets'+os.sep+'tweets'+str(self.page),'wb')
        # cPickle.dump(statuses,f,cPickle.HIGHEST_PROTOCOL);
        # f.close()
        f = open('twextract'+os.sep+'stored_tweets'+os.sep+'tweets'+str(self.page),'rb')
        statuses = cPickle.load(f)
        f.close()
        
        # - saves each tweet as a document into couchdb with following modifications (see below)
        for status in statuses:
            # - adds a 'doc_type' field to the json and sets it to 'raw_tweet'
            # - add a provider_namespace field (set it to 'twitter')
            # - add a provider_id field (set it to their twitter screenname)
            status['doc_type'] = 'raw_tweet'
            status['provider_namespace'] = 'twitter'
            status['provider_id'] = self.screen_name
            self.db.save(status)

n_pages = 3;
count = 10; # TODO: why is this only returning half as many as is requested?

def getTweets(screen_name='utunga',db_name='tweets'):
    api = tweepy.API(parser=tweepy.parsers.JSONParser())
    couch = couchdb.Server('http://127.0.0.1:5984')
    db = couch[db_name]
    
    # - given a twitter screen name (just using utunga for now)
    # - goes to twitter and downloads their *whole* timeline (just grabbing a few for now)
    
    # spawn a thread for each page
    q = Queue(n_pages)
    for page in xrange(1,n_pages+1):
        thread = StoreTweets(api,screen_name,page,db)
        thread.start()
        q.put(thread, True)
    
    # wait until all threads have finished 
    while not q.empty():
        thread = q.get(True)
        thread.join()


if __name__ == '__main__':
    # determine no. hits left before starting
    limit = tweepy.api.rate_limit_status()
    print 'start hits left: '+str(limit['remaining_hits'])
    print ''
    
    # store the specified users tweets into the specified DB
    getTweets();
    
    # determine no. hits left after completion
    limit = tweepy.api.rate_limit_status()
    print ''
    print 'end hits left: '+str(limit['remaining_hits'])
    print 'reset time: '+str(limit['reset_time'])