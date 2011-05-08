import couchdb

def setup():
    try:
        couchdb.Server('http://127.0.0.1:5984').create('hashmapd')
    except couchdb.PreconditionFailed:
        pass
