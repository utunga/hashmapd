"""
Tests the queue tasks in the twextract module
"""
import sys, os, time
from mock import Mock
# move the working dir up two levels so we can import hashmapd stuff
sys.path[0] = sys.path[0]+os.sep+'..'+os.sep+'..'

import cPickle
import gzip

import couchdb

import hashmapd

from twextract.request_queue import RequestQueue
from twextract.store_tweets import StoreTweets
import twextract.download_tweets as download_tweets

views = couchdb.Server('http://127.0.0.1:5984')['hashmapd']['_design/queue'];

# TODO: split this into a few test cases, and also make a failure test case
def test_downloads_and_stores_tweets_and_finally_requests_hash():
    # -- ARRANGE --
    # create factory to retrieve mock data
    factory = download_tweets.RetrieveTweetsFactory(use_mock_data=True);
    
    def next(queue_type):    
        if (download_tweets.Manager.no_requests < download_tweets.Manager.max_requests):
            download_tweets.Manager.no_requests += 1
            return Result(download_tweets.Manager.no_requests,{'id':str(download_tweets.Manager.no_requests),'username':'utunga','page':download_tweets.Manager.no_requests});
    
    def get_view_results(*args, **kwargs):
        view_results = {};
        view_results['utunga'] = []
        for i in xrange(download_tweets.Manager.max_requests-download_tweets.request_queue.completed_request.call_count):
             view_results['utunga'].append({'page':download_tweets.Manager.no_requests});
        return view_results
    
    class Result(object):
        def __init__(self,id,vals):
            self.id = str(id);
            self.values = vals;
            self.count = -1;
        
        def __getitem__(self,item):
            return self.values[item];
    
    # create mock object so that downloaded tweet values aren't stored
    download_tweets.store_tweets.store = Mock()
    # create mock object to request a few dummy tweets
    download_tweets.request_queue.next = Mock()
    download_tweets.request_queue.next.side_effect = next
    # create a mock object to record number of tweet download requests
    factory.retrieve_pickled_data = Mock()
    # create a mock object to ignore any db calls
    couchdb.client.Database.__getitem__ = Mock();
    download_tweets.request_queue.completed_request = Mock()
    download_tweets.request_queue.add_hash_request = Mock()
    # create a mock object to return the results of user view
    couchdb.client.Database.view = Mock();
    couchdb.client.Database.view.side_effect = get_view_results;
    
    # set up target
    manager = download_tweets.Manager('http://127.0.0.1:5984','hashmapd',factory,5);
    download_tweets.Manager.no_requests = 0
    download_tweets.Manager.max_requests = 2
    
    # -- ACT --
    manager.start()
    time.sleep(1.5)
    manager.exit()
    manager.join()
    
    # -- ASSERT --
    # ensure that the data was retrieved from twitter, and stored
    assert factory.retrieve_pickled_data.call_count == download_tweets.Manager.max_requests;
    assert download_tweets.store_tweets.store.call_count == download_tweets.Manager.max_requests;
    assert download_tweets.request_queue.add_hash_request.call_count == 1;

