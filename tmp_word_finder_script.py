import numpy as np
import codecs
from collections import defaultdict
from math import log
from hashmapd.common import debug

#from csv_unicode_helpers import UnicodeReader
from plistlib import Dict
from hashmapd.csv_unicode_helpers import UnicodeReader
#file = codecs.open('test.csv', encoding="utf-8",mode= "w")
#file.writelines(unicode("'@raphael" + unichr(432) +"'" +  "," + "'in'" +","+ str(10)))
#file.writelines(unicode("'@raphael'"+","+ "'for'" +","+ str(10))
#file.writelines(unicode("'@raphael'"+","+ "'unless'" +","+ str(10) +"\r\n"))
#file.writelines(unicode("'@raphael'" +","+ "'you'" +","+ str(10) ))
#file.close()

#arr=np.loadtxt(codecs.open('test.csv', encoding="utf-8",mode= "r"),delimiter=',')
#tuples = [(unicode('@raphael' + unichr(432)), unicode('in'),10),(unicode('@raphael'), unicode('for'),12), (unicode('@raphael'), unicode('in'),103)]

#arr = np.array(tuples, dtype=dt)
#print arr

debug("loading user totals..")
reader = UnicodeReader(open('projects/word_vectors/all_user_totals.csv'), quotechar="'")
overall_user_totals = {}
idx = 0
for [user, countstr] in reader:
    count = int(countstr)
    overall_user_totals[user] = count
    idx += 1
    if (idx % 1000==0):
        debug('.. %i rows'%idx)
debug('..done')


debug("loading token totals..")
reader = UnicodeReader(open('projects/word_vectors/all_token_counts.csv'), quotechar="'")
overall_token_totals = {}
idx = 0
for [token, countstr] in reader:
    count = int(countstr)
    overall_token_totals[token] = count
    idx += 1
    if (idx % 10000==0):
        debug('.. %i rows'%idx)
debug('..done')


#load user_token_counts.csv
tokens = []
users = []
user_ids = {}
token_ids = {}

debug("loading user token counts..")
reader = UnicodeReader(open('projects/word_vectors/all_user_token_counts.csv'), quotechar="'")
tmp = []
idx = 0
token_probabilities = defaultdict(list)
for [user, token, countstr] in reader:
    count = int(countstr)
    if (not token_ids.has_key(token)):
        tokens.append(token)
        token_ids[token] = len(tokens)-1

    if (not user_ids.has_key(user)):
        users.append(user)
        user_ids[user] = len(users)-1

    token_probabilities[token].append(count)
    token_user_prob = float(count) / overall_user_totals[user]
    tmp.append((user_ids[user], token_ids[token], count, token_user_prob))

    idx += 1
    if (idx % 100000==0):
        debug('.. %i rows'%idx)

user_token_counts = np.array(tmp)
debug('..done')



max_count =0


#print user_token_counts
#print user_overall_counts.items()
#print max_count

#find the max probability
max_prob = 0
max_prob_user = None
for [user, token, count, token_user_prob] in user_token_counts:
    #token = tokens[token_id]
    #user = users[user_id]
    max_prob = max(token_user_prob, max_prob)
    if max_prob==token_user_prob:
        max_prob_token=token
        max_prob_user = user



#class operations
# load data from file
# get ids, probabilties and max_prob
# iterate over user_token_counts

# out of class operiations
# 

# arrange bucket sizes so they are small for small values
# large for larger values
num_buckets=10

#buckets = [(1-log(x,10))*max_prob for x in xrange(num_buckets,0,-1)]
#buckets = [(ln(ln((x*max_prob)/num_buckets))) for x in xrange(num_buckets)]
#buckets = [0.0, 0.00015, 0.0002, 0.0005,0.001,0.002,0.005,0.01,0.02,0.05]
buckets = [0.0, 0.0002, 0.0005, 0.001, 0.002,0.005,0.01,0.02,0.05,0.2]

print buckets
#exit()

def bucket(val):
    for pos in xrange(len(buckets)-1):
        if buckets[pos+1]>val:
            return pos
    return len(buckets) -1

#probs = defaultdict(int)
probabilities = np.zeros((len(tokens), num_buckets), dtype=int)
for [user_id, token_id, count, token_user_prob] in user_token_counts:
    probabilities[token_id, bucket(token_user_prob)] += 1
#    probs[str(token_user_prob)] += 1

np.set_printoptions(threshold='nan')

# use this to print the actual distribution of 'probabilities'
#for token_user_prob in sorted(probs):
#    if (probs[token_user_prob] > 1):
#        print token_user_prob, probs[token_user_prob]

#print probabilities
#print token_probabilities.items()

#print max_prob_token, max_prob #, log(max_prob)
#print token_probabilities[max_prob_token]
#print user_token_counts[user_ids[max_prob_user]]

var_in_probs = probabilities.var(1)
best_tokens = {}
for i in xrange(len(tokens)):
    #print i
    #print tokens[i]
    #print var_in_probs[i-1]
    best_tokens[tokens[i]] = var_in_probs[i]

max_tokens = 5000
i=0
for token in sorted(best_tokens, key=best_tokens.get, reverse=True):
    print probabilities[token_ids[token]], token, best_tokens[token]
    i += 1
    if (i>max_tokens):
        break
  #    #arr.append(row)


#dt = np.dtype([('user', 'U4'),('token','U4'),('count','<i4')])
#file = codecs.open('all_user_token_counts.csv', encoding="utf-8",mode= "r")
#arr = np.genfromtxt(file, dtype=dt)
#arr = [] #np.array([], dtype=dt)
#for line in file.readlines():
#    tmp = line.split(',')
#    arr.append([tmp[0], tmp[1], int(tmp[2])])
#
#print arr[0]
#arr = np.array(arr, dtype=dt)
#print arr[0]
#arr = np.array([line.split(',') for line in file.readlines()], dtype=dt)
#print arr[0]
#file.close()
#print arr[:1]
#    print list(line.split(','));
#arr=np.loadtxt(,delimiter=',', dtype = dt)

#file = codecs.open('test.csv', encoding="utf-8",mode= "r")
#print(file.readline().split(','))
#print(file.readline())
#print(file.readline())
#print(file.readline())
#file.close()


# load from big, token_counts, user_counts
# pass through (on load) to get:
#   basic_token_spread
# filter by
#   user_thresholds (>min_word_count)
#   token_thresholds (>min_token_count, len(token)>min_token_length)
#   token_spread < max_token_spreads
#   gives new array? (re-ordered?)

# pass through to get:
#   user_token_prob
#   max_token_prob

# calc buckets
# pass through to get bucket probability frequencies
#   variance in frequency dist



# filter:


# calc
