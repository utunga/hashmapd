import couchdb

#==============================================================================
# Store a list of tweets in the specified db
#==============================================================================
class StoreTweets(object):
    
    def __init__(self):
        pass
    
    # - saves each tweet as a document into couchdb with the a few modifications
    def store(self, username, tweet_data, db):
        for tweet in tweet_data:
            # if this tweet is already stored in the db, ignore it (could happen
            # if a download is carried out twice for whatever reason)
            if tweet['id_str'] in db:
                return
            
            # - adds a 'doc_type' field to the json and sets it to 'raw_tweet'
            # - add a provider_namespace field (set it to 'twitter')
            # - add a provider_id field (set it to their twitter screenname)
            tweet['doc_type'] = 'raw_tweet'
            tweet['provider_namespace'] = 'twitter'
            tweet['provider_id'] = username
            db[tweet['id_str']] = tweet
            
            # if the user does not have a record stored in the db, make a
            # record for them (their hash, etc will later be stored here)
            if username not in db:
                self.store_user(username,tweet['user'],db);
            
            # if there are no more pending download requests for this user,
            # create a new hash request for the user
            # TODO: may want to optimize the decision here a bit more.
            # (eg: has the user had a hash calculated recently?,
            #      how much more data has been added for this user since last hash?,
            #      etc.)
            
            # TODO: create a view for the queued and underway download requests
            #       that filters by user / has username in the key
            
            # TODO: use the hash_request_queue class to handle add a new request
             
    
    def store_user(self, username, user_info, db):
        user_info['doc_type'] = 'user'
        user_info['hash'] = None
        db[username] = user_info;

