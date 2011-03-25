"""
Tests the queue tasks in the twextract module
"""
import os, sys
from mock import Mock
# move the working dir up two levels so we can import hashmapd stuff
sys.path[0] = sys.path[0]+os.sep+'..'+os.sep+'..'

import datetime, time
import couchdb

from twextract.tweet_request_queue import TweetRequestQueue

result1 = []

# test enqueue (add)
def test_enqueue():
    # variables used to test
    no_to_add = 5;

    # create a mock server
    couchdb.client.Database.save = Mock();
    couchdb.client.Database.save.side_effect = add_result1;
    
    # run the add method
    tweet_request_queue = TweetRequestQueue('http://127.0.0.1:5984','hashmapd');
    for i in xrange(no_to_add):
        tweet_request_queue.add('utunga',i);
    
    # ensure that the front of the queue remains unchanged, the back has been incremented, and the new values have been added
    for i in xrange(len(result1)):
        assert result1[i]['request_time'] != None
        assert result1[i]['username'] == 'utunga'
        assert result1[i]['page'] == i

def add_result1(value):
    result1.append(value);



result2 = []

# test enqueue screen name
def test_add_screen_name():
    # variables used to test
    no_to_add = 5;
    
    # create a mock server
    couchdb.client.Database.save = Mock();
    couchdb.client.Database.save.side_effect = add_result2;
    
    # run the add_screen_name method
    tweet_request_queue = TweetRequestQueue('http://127.0.0.1:5984','hashmapd');
    for i in xrange(no_to_add):
        tweet_request_queue.add_screen_name('utunga');
    
    # ensure that the front of the queue remains unchanged, the back has been incremented, and the new values have been added
    for i in xrange(no_to_add):
        for j in xrange(tweet_request_queue.n_pages):
            assert result2[tweet_request_queue.n_pages*i+j]['request_time'] != None
            assert result2[tweet_request_queue.n_pages*i+j]['username'] == 'utunga'
            assert result2[tweet_request_queue.n_pages*i+j]['page'] == (j+1)

def add_result2(value):
    result2.append(value);



dict = {}
result3 = []

# test dequeue (next)
def test_dequeue():
    # variables used to test
    no_to_remove = 5;
    
    # populate the queue with entries
    dict['_design/queue'] = couchdb.Server('http://127.0.0.1:5984')['hashmapd']['_design/queue']
    for i in xrange(no_to_remove):
        dict[str(i)] = {
            'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            'username':'utunga',
            'page':1,
            'type':'download_request'
        }
        time.sleep(0.01)
    
    # create a mock server and a mock views (that contain dummy values)
    couchdb.client.Database.__getitem__ = Mock();
    couchdb.client.Database.__getitem__.side_effect = get_from_db;
    couchdb.client.Database.__setitem__ = Mock();
    couchdb.client.Database.__setitem__.side_effect = store_in_db;
    couchdb.client.Database.query = Mock();
    couchdb.client.Database.query.side_effect = get_view_results;
    
    # run the remove method
    tweet_request_queue = TweetRequestQueue('http://127.0.0.1:5984','hashmapd');
    for i in xrange(no_to_remove):
        result3.append(tweet_request_queue.next());
    
    # ensure that the requests are popped off the queue in the correct order
    for i in xrange(no_to_remove-1):
        assert result3[i]['request_time'] < result3[i+1]['request_time']


def get_from_db(item):
    return dict[item]

def store_in_db(key,value):
    dict[key] = value
    
def get_view_results(view):
    if (view == dict['_design/queue']['views']['underway_download_requests']['map']):
        view_results = []
        for k,v in dict.iteritems():
            if k != '_design/queue':
                try:
                    view_results.append(
                        couchdb.client.Row(id=str(k),key=v['started_time'],
                            value={'id':str(k),
                                'username':'utunga',
                                'page':1,
                                'type':'download_request'})
                        )
                except KeyError:
                    pass
        view_results.sort()
        return view_results
    
    elif (view == dict['_design/queue']['views']['queued_download_requests']['map']):
        view_results = []
        for k,v in dict.iteritems():
            if k != '_design/queue' and 'started_time' not in v:
                view_results.append(
                    couchdb.client.Row(id=str(k),key=v['request_time'],
                        value={'id':str(k),
                            'username':'utunga',
                            'page':1,
                            'type':'download_request'})
                    )
        view_results.sort()
        return view_results
    
def compare(a,b):
    return cmp(a['request_time'],b['request_time'])
