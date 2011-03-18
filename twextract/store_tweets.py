import couchdb

#==============================================================================
# Store a list of tweets in the specified db
#==============================================================================
class StoreTweets(object):
    
    def __init__(self):
        pass
    
    # - saves each tweet as a document into couchdb with the a few modifications
    def store(self, screen_name, tweet_data, db):
        for tweet in tweet_data:
            # - adds a 'doc_type' field to the json and sets it to 'raw_tweet'
            # - add a provider_namespace field (set it to 'twitter')
            # - add a provider_id field (set it to their twitter screenname)
            tweet['doc_type'] = 'raw_tweet'
            tweet['provider_namespace'] = 'twitter'
            tweet['provider_id'] = screen_name
            db.save(tweet)

