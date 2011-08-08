import gzip
import json
import os
import urllib2
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S', logfile='/ebs/log/export_tweets.log')
ALL_DOCS = 'http://localhost:5984/hashmapd/_all_docs'
BASE_DIR = '/ebs/data/tweets/'

def emit(username, tweet):
   directory = os.path.join(BASE_DIR, username[0])
   try:
       os.makedirs(directory)
   except OSError:
       pass
   f = gzip.open(os.path.join(directory, username+'.gz'), mode='ab')
   json.dump(tweet, f)
   f.write('\n')
   f.close()

#We loop over all tweets writing them to file
startkey = 'tweet_'
go = True
ids = []
MAX_IDS = 10000
count = 0
while go:
   url = ALL_DOCS+'?startkey="%s"&include_docs=true&limit=1000'%(startkey,)
   lines = urllib2.urlopen(url).read()
   rows = json.loads(lines)['rows']
   if not len(rows):
      go = False
      break
   for row in rows:
       if row["id"].startswith("tweet_"):  #looks like a tweet
           if row["id"] not in ids:
               ids.insert(0, row["id"])
               if len(ids) > MAX_IDS:
	           ids.pop()
               tweet = row['doc']
               startkey = row['key']
               try:
                   if tweet.has_key('username'):
                      username = tweet['username']
                   else:
                      username = tweet['screen_name']
                   count += 1
                   emit(username, tweet)
               except KeyError:
                   logging.error("Keyerror: " + str(row))
   logging.info('%s, %s'%(count, startkey))
