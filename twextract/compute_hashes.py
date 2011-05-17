import os
import sys
import getopt
import time
from Queue import Queue
import threading
import csv
import cPickle
import gzip
import datetime

import numpy
from mock import Mock
import couchdb
import theano

import hashmapd
import run
from twextract.request_queue import RequestQueue
from twextract.store_results import StoreResults


request_queue = RequestQueue()
store_results = StoreResults()


class ComputeHash(threading.Thread):
    """
    Computes a hash given the specified raw word counts, and stores the result in
    under the users information in the specified db
    """
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
            hash = self.compute_hash(self.raw_word_counts)
            # store the tweets in specified db
            store_results.store_hash(self.screen_name,hash[0],self.db)
            # TODO: also do TSNE and compute co-ordinates here
            coords = self.calc_tsne(hash,cfg.tsne.desired_dims,cfg.tsne.perplexity,cfg.tsne.pca_dims)
            # store the co-ordinates in specified db
            store_results.store_coords(self.screen_name,coords[0],self.db)
            
        except Exception, err:
            # notify manager of error
            self.manager.notify_failed(self,err)
            return
        
        # notify manager that data has been retrieved
        self.manager.notify_completed(self)
    
    def compute_hash(self,raw_word_counts):
        return run.get_output_codes(self.manager.smh,raw_word_counts)
    
    def calc_tsne(self,hash,desired_dims,perplexity,pca_dims):
        return run.calc_tsne(hash,desired_dims,perplexity,pca_dims)


class Manager(threading.Thread):
    """
    Loop until user terminates program. Obtain hash requests from the queue and
    spawn worker threads to compute hash codes.
 
    Blocks when the maximum number of simultanous requests are underway.
    Currently busy-waits when there are no requests on the queue.
    """
    
    def __init__(self, server_url='http://127.0.0.1:5984', db_name='hashmapd',\
                    max_simultaneous_requests=5):
        threading.Thread.__init__(self)
        self.db = couchdb.Server(server_url)[db_name]
        self.worker_threads = []
        self.semaphore = semaphore = threading.Semaphore(max_simultaneous_requests)
        self.terminate = False
        # load the SMH in once, and use to compute all hashes
        self.words_dict = load_words_dict()
        self.smh = run.load_model(cfg.train.cost, n_ins=cfg.shape.input_vector_length,\
                       mid_layer_sizes=list(cfg.shape.mid_layer_sizes), inner_code_length=cfg.shape.inner_code_length,\
                       weights_file=cfg.train.weights_file)
    
    def notify_completed(self,thread):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        
        # write finished time
        row = self.db[thread.request_id]
        request_queue.completed_request(row,thread.request_id)
        # print notification of completion
        print 'Computed hash ('+str(thread.screen_name)+')'
    
    def notify_failed(self,thread,err):
        self.worker_threads.remove(thread)
        self.semaphore.release()
        # clear started time, so that the job will be restarted
        # TODO: (is this a good idea? what if the job keeps failing over and over for some reason?)  
        row = self.db[thread.request_id]
        request_queue.failed_request(row,thread.request_id)
        # print error message
        print >> sys.stderr, 'Error computing hash/coords ('+str(thread.screen_name)+'):\n'+str(err)
    
    def run(self):
        # - obtain a username from db that needs hash computed
        # spawn a thread for each job
        while self.terminate == False:
            # get the next request
            next_request = request_queue.next('hash')
            # if the queue is empty, check for new data (every 2 seconds for now)
            # (might want to just terminate here)
            if next_request == None:
                time.sleep(0.05)
                continue
            
            screen_name = next_request['username']
            request_id = next_request.id
            
            # construct histogram based on the user's tweets
            visible_data = self.construct_histogram(screen_name)
            # convert from sums to percentages
            visible_data = visible_data/visible_data.sum()
            
            # acquire a lock before creating new thread
            self.semaphore.acquire()
            # spawn a worker thread to compute a hash for the specified user 
            thread = ComputeHash(self,screen_name,visible_data,self.db,request_id)
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
        results = self.db.view('tweets/tokenize', reduce=True, group=True) 
        # obtain results from this user only
        results = results[[screen_name]:[screen_name,'Z']]
        
        # build the histogram of relevant word counts for this user
        visible_data = numpy.zeros((1,cfg.shape.input_vector_length),dtype=theano.config.floatX)
        # TODO: this a moderately expensive operation. consider trying to make
        #       it more efficient at some point
        for result in results:
            try:
                visible_data[0,self.words_dict[result.key[1]]] = result.value
            except KeyError:
                pass
        
        return visible_data
    
    def exit(self):
        self.terminate = True

def load_words_dict():
    # load in the list of words used in trained SMH (throw away header line)
    csvReader = csv.reader(open(cfg.input.input_words,"rb"),delimiter=',',quotechar='\"')
    csvReader.next()
    
    # load data into dictionary for fast access 
    words_dict = {}
    for line in csvReader:
        words_dict[line[1].lower()] = int(line[0])
    return words_dict


def usage():
    return 'usage: get_tweets                                                   \n'+\
           '   [-c config]        specifies config file to load                 '

if __name__ == '__main__':
    """
    Main method to intialize and run the Manager indefinitely
    """
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
    
    # run the man))
    manager = Manager(cfg.raw.couch_server_url,cfg.raw.couch_db,cfg.raw.max_simultaneous_requests)
    manager.start()
    
    # keep running until user types 'exit', then terminate nicely
    # (ie: allow all currently running jobs to complete before closing)
    print ''
    while True:
        input = raw_input("type \'exit\' to terminate:\n\n")
        if input == 'exit':
            manager.exit()
            print 'terminating ...'
            break



