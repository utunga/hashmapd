import couchdb
import json

couch = couchdb.Server('http://localhost:5984')
db = couch['frontend_dev']

xbox = json.load(file('xbox.json'))
for row in xbox['rows']:
    doc = row['value']
    db[doc['_id']] = doc
    print doc['_id']
