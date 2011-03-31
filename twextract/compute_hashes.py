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
import numpy

import couchdb

import theano

import hashmapd
import run
from twextract.tweet_request_queue import TweetRequestQueue
from twextract.store_hashes import StoreHashes


tweet_request_queue = TweetRequestQueue();
store_hashes = StoreHashes();


#==============================================================================
# Factory to produce ComputeHash worker threads
#==============================================================================
class ComputeHashFactory(object):
    
    def __init__(self,use_mock_data=False,save_mock_data=False):
        self.use_mock_data = use_mock_data;
     
    def get_worker_thread(self,manager,screen_name,raw_word_counts,db,request_id):
        thread = ComputeHash(manager,screen_name,raw_word_counts,db,request_id)
        
        if (self.use_mock_data):
            thread.compute_hash = Mock()
            thread.compute_hash.side_effect = self.compute_fake_hash
        
        return thread
    
    def compute_fake_hash(thread,raw_word_counts):
        return numpy.random((1,cfg.shape.inner_code_length),dtype=theano.config.floatX);


#==============================================================================
# Computes a hash given the specified raw word counts, and stores the result in
# under the users information in the specified db
#==============================================================================
class ComputeHash(threading.Thread):
    
    def __init__(self,manager,screen_name,raw_word_counts,db,request_id):
        threading.Thread.__init__(self)
        self.manager = manager
        self.screen_name = screen_name
        self.raw_word_counts = raw_word_counts
        self.db = db        
        self.request_id = request_id
    
    def run(self):
        try:
            # obtain hash
            hash = self.compute_hash(self.raw_word_counts);
            # store the tweets in specified db
            store_hashes.store(self.screen_name,hash,self.db)
            # TODO: could also do TSNE and compute co-ordinates here
            
        except Exception, err:
            # notify manager of error
            self.manager.notifyFailed(self,err)
            return
        
        # notify manager that data has been retrieved
        self.manager.notifyCompleted(self)
    
    def compute_hash(self,raw_word_counts):
        return run.get_output_codes(self.manager.smh,raw_word_counts)[0];

import time

#==============================================================================
# Loop until user terminates program. Obtain hash requests from the queue and
# spawn worker threads to compute hash codes.
# 
# Blocks when the maximum number of simultanous requests are underway.
# Currently busy-waits when there are no requests on the queue.
#==============================================================================
class Manager(threading.Thread):
    
    def __init__(self, server_url='http://127.0.0.1:5984', db_name='hashmapd',\
            retrieval_factory=ComputeHashFactory(), max_simultaneous_requests=5):
        threading.Thread.__init__(self)
        self.db = couchdb.Server(server_url)[db_name]
        self.retrieval_factory = retrieval_factory
        self.worker_threads = []
        self.semaphore = semaphore = threading.Semaphore(max_simultaneous_requests)
        self.terminate = False
        # load the SMH in once, and use to compute all hashes
        self.smh = run.load_model(cfg.train.cost, n_ins=cfg.shape.input_vector_length,\
                       mid_layer_sizes=list(cfg.shape.mid_layer_sizes), inner_code_length=cfg.shape.inner_code_length,\
                       weights_file=cfg.train.weights_file)
    
    def notifyCompleted(self,thread):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        
        # write finished time
        row = self.db[thread.request_id]
        tweet_request_queue.completed_request(row,thread.request_id);
        # print notification of completion
        print 'Computed hash ('+str(thread.screen_name)+')'
    
    def notifyFailed(self,thread,err):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # clear started time, so that the job will be restarted
        # TODO: (is this a good idea? what if the job keeps failing over and over for some reason?)  
        row = self.db[thread.request_id]
        tweet_request_queue.failed_request(row,thread.request_id);
        # print error message
        print >> sys.stderr, 'Error computing hash ('+str(thread.screen_name)+'):\n'+str(err)
    
    def run(self):
        # - obtain a username from db that needs hash computed
        # spawn a thread for each job
        while self.terminate == False:
            # get the next request
            next_request = tweet_request_queue.next('hash')
            # if the queue is empty, check for new data (every 2 seconds for now)
            # (might want to just terminate here)
            if next_request == None:
                time.sleep(2)
                continue;
            
            screen_name = next_request['username']
            request_id = next_request.id
            
            # acquire a lock before creating new thread
            self.semaphore.acquire()
            # spawn a worker thread to compute a hash for the specified user
            visible_data = self.construct_histogram(screen_name)
            thread = self.retrieval_factory.get_worker_thread(self,screen_name,visible_data,self.db,request_id)
            self.worker_threads.append(thread)
            thread.start()
        
        # wait until all threads have finished 
        while len(self.worker_threads) > 0:
            thread = self.worker_threads[0]
            thread.join()
        
        # determine no hits left after completion
        print ''
        print 'exited compute_hashes.py'
    
    def construct_histogram(self,screen_name):        
        # obtain results from view 
        tokenized_view = self.db['_design/tweets']['views']['tokenize'];
        map_view = tokenized_view['map'];
        reduce_view = tokenized_view['reduce'];
        results = self.db.query(map_view,reduce_fun=reduce_view,group=True)
        # obtain results from this user only
        results = results[[screen_name]:[screen_name,'Z']]
        
        # TODO: need to get an list of words that we are using to
        #       evaluate tweets, so that we can build a histogram 
        words_dict = {'world':0,'you':2999}
        visible_data = numpy.zeros((1,cfg.shape.input_vector_length),dtype=theano.config.floatX);
        
        # TODO: this a moderately expensive operation. consider trying to make
        #       it more efficient at some point
        for result in results:
            try:
                visible_data[0,words_dict[result.key[1]]] = result.value
            except:
                pass
        
        return visible_data
    
    def exit(self):
        self.terminate = True;


#==============================================================================
# Main method to intialize and run the Manager indefinitely
#==============================================================================

def usage():
    return 'usage: get_tweets                                                   \n'+\
           '   [-c config]        specifies config file to load                 '

if __name__ == '__main__':
    print 'starting compute_hashes.py'
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
            cfg = hashmapd.LoadConfig(a)
        else:
            assert False, "unhandled option"
    
    if cfg is None:
        cfg = hashmapd.DefaultConfig()
    
    # run the manager
    factory = ComputeHashFactory(use_mock_data=False);
    thread = Manager(cfg.raw.couch_server_url,cfg.raw.couch_db,factory,cfg.raw.max_simultaneous_requests);
    thread.start()
    
    # keep running until user types 'exit', then terminate nicely
    # (ie: allow all currently running jobs to complete before closing)
    print ''
    while True:
        input = raw_input("type \'exit\' to terminate:\n\n");
        if input == 'exit':
            thread.exit();
            print 'terminating ...'
            break



