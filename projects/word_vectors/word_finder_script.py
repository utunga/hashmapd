import numpy as np
import codecs
from collections import defaultdict
from math import log
from hashmapd.common import debug
from hashmapd.token_counts import TokenCounts

min_token_count=10 #minimum number of times a token must have been used (across all users)
skip_common_tokens_cutoff=0.0001 #skip the top 0.01% most common words
min_user_total=30 #minimum number of words per user (drop users under this threshold)
max_percent_cover=0.95 #maximum fraction of users to use a token at least once (drop words above this threshold)
min_token_len = 3 # drop tokens shorter than this
number_tokens_wanted = 20000
data_dir = 'projects/word_vectors/raw/amdata'

# before we even analyze for entropy etc, we drop
# some edge cases tokens (eg 'the') - this saves
# a lot on amount of memory/time to process and we know from test these words
# have a low entropy anyway (so won't make the cut)
def pre_token_filter(token):
    if (len(token)<min_token_len):
        return False
    if (data.token_totals[token] < min_token_count):
        return False
    if (data.percent_cover(token)>max_percent_cover):
        return False
    return True

def user_filter(user):
    if (data.user_totals[user]<min_user_total):
        return False
    return True

def token_filter_by_variance(token):
    return (token in included_tokens)


# ------------------------------------------------------------
# main program starts here

data = TokenCounts(data_dir = data_dir)
data.load_from_csv(min_token_count=min_token_count, skip_common_tokens_cutoff=skip_common_tokens_cutoff, min_user_total=min_user_total)

data.write_to_csv(file_prefix='stage0_', token_filter_fun=pre_token_filter, user_filter_fun=user_filter)
data.load_from_csv(file_prefix='stage0_')
# data.user_token_counts is a list, each element being a tuple of:
# (user_id, token_id, count, token_user_prob)
# token_user_prob is prob(token | user), so these sum to 1 per user.


probs = np.zeros((len(data.user_token_counts)), dtype=float)
for i,[user_id, token_id, count, token_user_prob] in enumerate(data.user_token_counts):
    probs[i] = token_user_prob
    # But isn't there some vastly more efficient way to do this?!
probs = np.sort(probs)
print 'THE FIRST TEN: ',probs[0:10]
print 'THE LAST TEN: ',probs[-10:-1]

num_buckets=10
step = len(probs)/num_buckets
buckets=[]
print 'len: ',len(probs)
for b in range(num_buckets):
    buckets.append(probs[b*step])
print 'Buckets are: ',buckets    


# arrange bucket sizes so they are small for small values
# large for larger values

histogram_counts = np.zeros((data.num_tokens, num_buckets), dtype=int)

# put words into buckets.
def bucket(val):
    """ returns the index of the bucket that val is in """
    for pos in xrange(len(buckets)-1):
        if buckets[pos+1]>val:
            return pos
    return len(buckets) -1

for [user_id, token_id, count, token_user_prob] in data.user_token_counts:
    histogram_counts[token_id, bucket(token_user_prob)] += 1

print 'histogram_counts shape is ',histogram_counts.shape
epsilon = 0.00000001 # a tiny prob to stop the 'never seen' probablity being zero!
normed_histogram = np.transpose(np.transpose(histogram_counts+epsilon) / (1.0*np.sum(histogram_counts+epsilon,1)))
histogram_entropy = np.sum(-normed_histogram * np.log(normed_histogram),1)

var_in_probs = histogram_counts.var(1)
best_tokens = {}
for i in xrange(data.num_tokens):
    best_tokens[data.tokens[i]] = var_in_probs[i]  * (1- data.percent_cover(data.tokens[i]))

very_best_tokens = {}
for i in xrange(data.num_tokens):
    very_best_tokens[data.tokens[i]] = histogram_entropy[i] # * (1- data.percent_cover(data.tokens[i]))


sorted_tokens = sorted(very_best_tokens, key=very_best_tokens.get, reverse=True)
included_tokens = sorted_tokens[0:number_tokens_wanted]

data.write_to_csv('inc_', token_filter_fun=token_filter_by_variance)


# the rest is just debug - but it'd be nice to write this info to a file
np.set_printoptions(threshold='nan')
max_tokens = number_tokens_wanted
i=0
print '           token, variance, percent_cover, entropy'
for token in included_tokens:
    print '%20s  %.5f  %.5f   %.5f \t' % (token, best_tokens[token], data.percent_cover(token), very_best_tokens[token]),
    print histogram_counts[data.token_ids[token]]
    i += 1
    if (i>max_tokens):
        break
