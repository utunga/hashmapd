"""
Tests the queue tasks in the twextract module
"""
import sys, os
from mock import Mock

import cPickle
import gzip

import couchdb

from twextract.tweet_request_queue import TweetRequestQueue
from twextract.store_tweets import StoreTweets
import twextract.download_tweets as download_tweets

# test enqueue and dequeue
def test_downloads():
    # load the default config, and run the retrieve tweets thread with a mock object to return dummy data
    download_tweets.store_tweets.store = Mock()
    thread = download_tweets.RetrieveTweets(Mock(),'utunga',1,couchdb.Server('http://127.0.0.1:5984')['hashmapd'],1)
    # for now, we just load stored stored from a pickled file (don't want to actually do the request) 
    thread.retrieve_data_from_twitter = Mock();
    thread.retrieve_data_from_twitter.return_value = retrieve_pickled_data('utunga',1)
    thread.start()
    thread.join()
    
    # ensure that the data was retrieved from twitter once, and stored once, and that the correct number of docs were stored
    assert thread.retrieve_data_from_twitter.call_count == 1;
    assert download_tweets.store_tweets.store.call_count == 1;
    assert len(download_tweets.store_tweets.store.call_args[0][1]) == download_tweets.count;


def retrieve_pickled_data(username,page):
    f = open(sys.path[0]+os.sep+'twextract'+os.sep+'stored_tweets'+os.sep+str(username)+str(page),'rb')
    tweet_data = cPickle.load(f)
    f.close()
    return tweet_data


