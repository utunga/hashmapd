import couchdb

class StoreTweets(object):
    """
    Store a list of tweets in the specified db
    """
    def __init__(self):
        pass
    
    # - saves each tweet as a document into couchdb with the a few modifications
    def store(self, username, tweet_data, db):
        for tweet in tweet_data:
            key = 'tweet_%s_%s'%(username, tweet['id_str'])
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
                db[key] = tweet
            except couchdb.ResourceConflict:
                old_tweet = db[key]
                tweet['_rev'] = old_tweet['_rev']
                db[key] = tweet
                return

