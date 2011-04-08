"""
Tests the queue tasks in the twextract module
"""
import os, sys
from mock import Mock

import datetime, time
import couchdb

import hashmapd

from twextract.request_queue import RequestQueue

# test enqueue (add)
def test_queue_adds_request():
    # -- ARRANGE --
    # variables used to test
    no_to_add = 5
    
    # set up result
    result = []
    def add_result(key,value):
        result.append(value)
    
    # create a mock server
    couchdb.client.Database.__setitem__ = Mock()
    couchdb.client.Database.__setitem__.side_effect = add_result
    
    #set up target
    request_queue = RequestQueue('http://127.0.0.1:5984','hashmapd')
    
    # -- ACT --
    for i in xrange(no_to_add):
        request_queue.add_download_request('utunga',i)
    
    # -- ASSERT --
    # ensure that the front of the queue remains unchanged, the back has been incremented, and the new values have been added
    for i in xrange(len(result)):
        assert result[i]['request_time'] != None
        assert result[i]['username'] == 'utunga'
        assert result[i]['page'] == i



# test enqueue (add) screen name
def test_queue_adds_requests_for_username():
    # -- ARRANGE --
    # variables used to test
    no_to_add = 5
    
    # set up result
    result = []
    def add_result(key,value):
        result.append(value)
    
    # create a mock server
    couchdb.client.Database.__setitem__ = Mock()
    couchdb.client.Database.__setitem__.side_effect = add_result
    
    # set up target
    request_queue = RequestQueue('http://127.0.0.1:5984','hashmapd')
    
    # -- ACT --
    for i in xrange(no_to_add):
        request_queue.add_download_requests_for_username('utunga')
    
    # -- ASSERT --
    # ensure that the front of the queue remains unchanged, the back has been incremented, and the new values have been added
    for i in xrange(no_to_add):
        for j in xrange(request_queue.n_pages):
            assert result[request_queue.n_pages*i+j]['request_time'] != None
            assert result[request_queue.n_pages*i+j]['username'] == 'utunga'
            assert result[request_queue.n_pages*i+j]['page'] == (j+1)



# test dequeue (next)
def test_download_requests_dequeued_in_order():
    # -- ARRANGE --
    # variables used to test
    no_to_remove = 5
    
    # set up result
    dict = {}
    result = []
    
    def get_from_db(item):
        return dict[item]
    
    def store_in_db(key,value):
        dict[key] = value
    
    # returns a list of values taken from the dumnmy values dictionary, that are 
    # formatted and ordered in the same way as the relevant view would be
    def get_view_results(view,reduce):
        if (view == 'queue/queued_download_requests'):
            return generate_view('queued','download')
        elif (view == 'queue/underway_download_requests'):
            return generate_view('underway','download')
        elif (view == 'queue/queued_hash_requests'):
            return generate_view('queued','hash')
        elif (view == 'queue/underway_hash_requests'):
            return generate_view('underway','hash')
    
    def generate_view(doc_status,queue_name):
        if doc_status == 'queued':
            view_results = []
            for k,v in dict.iteritems():
                if k != '_design/queue' and 'started_time' not in v:
                    view_results.append(
                        couchdb.client.Row(id=str(k),key=v['request_time'],
                            value={'id':str(k),
                                'username':'utunga',
                                'page':1,
                                'type':queue_name+'_request'})
                        )
            view_results.sort()
            return view_results
        elif doc_status == 'underway':
            view_results = []
            for k,v in dict.iteritems():
                if k != '_design/queue':
                    try:
                        view_results.append(
                            couchdb.client.Row(id=str(k),key=v['started_time'],
                                value={'id':str(k),
                                    'username':'utunga',
                                    'page':1,
                                    'type':queue_name+'_request'})
                            )
                    except KeyError:
                        pass
            view_results.sort()
            return view_results
    
    # populate the mock queue with entries
    for i in xrange(no_to_remove):
        dict[str(i)] = {
            'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            'username':'utunga',
            'page':1,
            'type':'download_request'
        }
        time.sleep(0.01)
    
    # create a mock server and mock views (that contain dummy values)
    couchdb.client.Database.__getitem__ = Mock()
    couchdb.client.Database.__getitem__.side_effect = get_from_db
    couchdb.client.Database.__setitem__ = Mock()
    couchdb.client.Database.__setitem__.side_effect = store_in_db
    couchdb.client.Database.view = Mock()
    couchdb.client.Database.view.side_effect = get_view_results
    
    # set up target
    request_queue = RequestQueue('http://127.0.0.1:5984','hashmapd')
    
    # -- ACT --
    for i in xrange(no_to_remove):
        result.append(request_queue.next('download'))
    
    # -- ASSERT --
    # ensure that the requests were popped off the queue in the correct order
    for i in xrange(no_to_remove-1):
        assert result[i]['request_time'] < result[i+1]['request_time']
    

