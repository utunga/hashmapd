"""
Tests the queue tasks in the twextract module
"""
import sys, os, time
from mock import Mock

import cPickle
import gzip

import couchdb

import hashmapd

import tweepy
from twextract.request_queue import RequestQueue
from twextract.store_tweets import StoreTweets
import twextract.download_tweets as download_tweets


views = couchdb.Server('http://127.0.0.1:5984')['hashmapd']['_design/queue']

# TODO: split this into a few test cases, and also make a failure test case
def test_downloads_and_stores_tweets_and_user_and_requests_hash():
    # -- ARRANGE --
    def next(queue_type):
        if (download_tweets.Manager.no_requests < download_tweets.Manager.max_requests):
            download_tweets.Manager.no_requests += 1
            return Result(download_tweets.Manager.no_requests,{'id':str(download_tweets.Manager.no_requests),'username':'utunga','page':download_tweets.Manager.no_requests})
        
    def contains(username):
        return (download_tweets.request_queue.next.call_count == download_tweets.Manager.max_requests-1)
    
    def get_view_results(*args, **kwargs):
        view_results = {}
        view_results['utunga'] = []
        for i in xrange(download_tweets.Manager.max_requests-download_tweets.request_queue.completed_request.call_count):
             view_results['utunga'].append({'page':download_tweets.Manager.no_requests})
        return view_results
    
    class Result(object):
        def __init__(self,id,vals):
            self.id = str(id)
            self.values = vals
            self.count = -1
        
        def __getitem__(self,item):
            return self.values[item]
    
    # create a mock object to block api calls
    download_tweets.api = Mock()
    download_tweets.api.rate_limit_status = Mock()
    download_tweets.api.rate_limit_status.return_value = {'remaining_hits':100,'reset_time':'N/A'}
    # create a mock object to return pickled data instead of downloading from twitter 
    download_tweets.RetrieveTweets.retrieve_data_from_twitter = Mock()
    download_tweets.RetrieveTweets.retrieve_data_from_twitter.side_effect = retrieve_pickled_data
    # create a mock object to block calls to store user
    download_tweets.StoreUser.run = Mock()
    # create mock object so that downloaded tweet values aren't stored
    download_tweets.store_tweets.store = Mock()
    # create mock object to request a few dummy tweets
    download_tweets.request_queue.next = Mock()
    download_tweets.request_queue.next.side_effect = next
    # create a mock object to ignore any db calls
    couchdb.client.Database.__getitem__ = Mock()
    couchdb.client.Database.__contains__ = Mock()
    couchdb.client.Database.__contains__.side_effect = contains
    download_tweets.request_queue.completed_request = Mock()
    download_tweets.request_queue.add_hash_request = Mock()
    # create a mock object to return the results of user view
    couchdb.client.Database.view = Mock()
    couchdb.client.Database.view.side_effect = get_view_results
    
    # set up target
    manager = download_tweets.Manager('http://127.0.0.1:5984','hashmapd',5)
    download_tweets.Manager.no_requests = 0
    download_tweets.Manager.max_requests = 2
    
    # -- ACT --
    manager.start()
    time.sleep(0.1)
    manager.exit()
    manager.join()
    
    # -- ASSERT --
    # ensure that the data was retrieved from twitter, and stored
    assert download_tweets.RetrieveTweets.retrieve_data_from_twitter.call_count == download_tweets.Manager.max_requests
    assert download_tweets.store_tweets.store.call_count == download_tweets.Manager.max_requests
    assert download_tweets.StoreUser.run.call_count == 1
    assert download_tweets.request_queue.add_hash_request.call_count == 1


def retrieve_pickled_data(screen_name,page):
    f = open(sys.path[0]+os.sep+'twextract'+os.sep+'stored_tweets'+os.sep+str(screen_name)+str(page),'rb')
    tweet_data = cPickle.load(f)
    f.close()
    return tweet_data

def pickle_data(screen_name,page):
    tweet_data = api.user_timeline(screen_name=screen_name,page=page,count=count,include_entities=True,trim_user=True,include_rts=False)
    f = open(sys.path[0]+os.sep+'twextract'+os.sep+'stored_tweets'+os.sep+str(screen_name)+str(page),'wb')
    cPickle.dump(tweet_data,f,cPickle.HIGHEST_PROTOCOL)
    f.close()
    return tweet_data
