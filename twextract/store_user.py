import os
import sys
import threading

import couchdb
import tweepy
from tweepy.error import TweepError

class StoreUser(threading.Thread):
    """
    Store a new entry for the user
    """
    def __init__(self,username,db,tweepy_api):
        threading.Thread.__init__(self)
        self.username = username
        self.db = db
        self.api = tweepy_api
    
    # make a record for the given user (their hash, etc will later be
    # stored here)
    def run(self):
        try:
            self.db['twuser_'+self.username] = {'loading':''}
        except couchdb.ResourceConflict:
            return
        
        try:
            self.store_user(self.api.get_user(screen_name=self.username),self.db['twuser_'+self.username]['_rev'])
        except couchdb.ResourceConflict:
            del self.db['twuser_'+self.username]
	except TweepError, err:
            if str(err) == 'Not found':
                del self.db['twuser_'+self.username]
                 
    
    def store_user(self,user_info,rev):
        user_info['doc_type'] = 'user'
        user_info['provider_namespace'] = 'twitter'
        user_info['hash'] = None
        user_info['coords'] = None
        user_info['_rev'] = rev
        self.db['twuser_'+self.username] = user_info

