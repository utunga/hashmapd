import couchdb

#==============================================================================
# Store the calculated hash in the specified db
#==============================================================================
class StoreHashes(object):
    
    def __init__(self):
        pass
    
    def store(self, username, hash, db):
        # store the hash in the db under this user
        try:
            row = db[username];
            row['hash'] = str(hash);
            db[username] = row;
            
        except couchdb.ResourceConflict:
            return
    
    
