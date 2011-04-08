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
            # - adds a 'doc_type' field to the json and sets it to 'raw_tweet'
            # - add a provider_namespace field (set it to 'twitter')
            # - add a provider_id field (set it to their twitter screenname)
            tweet['doc_type'] = 'raw_tweet'
            tweet['username'] = username
            tweet['provider_namespace'] = 'twitter'
            
            # store the tweet in the db (if this tweet was already stored in 
            # the db, replace it with the new version - this could happen if a 
            # download is carried out twice for whatever reason)
            try:
                db['tweet_'+tweet['id_str']] = tweet
            except couchdb.ResourceConflict:
                old_tweet = db['tweet_'+tweet['id_str']]
                tweet['_rev'] = old_tweet['_rev']
                db['tweet_'+tweet['id_str']] = tweet
                return

