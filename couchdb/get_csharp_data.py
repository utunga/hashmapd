import os
import sys
    
import couchdb

server = couchdb.Server('http://localhost:5984')
csharpDB = server['replicated']
db = server['hashmapd']

# push tweets
results = csharpDB.view('get_csharp_tweets/get_tweets', reduce=False)
for result in results:
    db[result.key] = result.value

# push users and hash request
results = csharpDB.view('get_csharp_tweets/get_users', reduce=True, group=True)
for result in results:
    print result
    db["twuser_"+result.key[0]] = result.key[1]
    
    # ensure there's no other download requests for this user, then add a hash request
    d_requests = self.db.view('queue/queued_user_download_requests', reduce=False)
    if len(d_requests[result.key[0]]) == 0:
        self.db.save({'username':result.key[0],\
          'request_time':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),'doc_type':'hash_request'})
