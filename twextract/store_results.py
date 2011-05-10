import couchdb

class StoreResults(object):
    """
    Store the calculated hash or coords in the specified db
    """
    def __init__(self):
        pass
    
    def store_hash(self, username, hash, db):
        # store the hash in the db under this user
        try:
            row = db[username]
            row['hash'] = str(hash)
            db[username] = row
            
        except couchdb.ResourceConflict:
            return
    
    def store_coords(self, username, coords, db):
        # store the hash in the db under this user
        try:
            row = db[username]
            row['coords'] = str(coords)
            db[username] = row
            
        except couchdb.ResourceConflict:
            return
    
