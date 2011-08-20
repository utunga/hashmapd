import numpy as np
import codecs
from collections import defaultdict
from math import log
from hashmapd.common import debug
from hashmapd.token_counts import TokenCounts


data = TokenCounts(data_dir = 'projects/word_vectors/')
data.load_from_csv()

#find the max probability

def token_filter(token):
    if (len(token)<4):
        return False

    if (data.token_totals[token] < 5):
        return False

    if (data.percent_cover(token)>.5):
        return False

    return True

def user_filter(user):
    if (data.user_totals[user]<100):
        return False
    return True

data.write_to_csv(file_prefix='stage0_', token_filter_fun=token_filter, user_filter_fun=user_filter)
data.load_from_csv(file_prefix='stage0_')

#max_prob = np.max(data.user_token_counts, axis=0)[3]
#print max_prob

# arrange bucket sizes so they are small for small values
# large for larger values
num_buckets=10
buckets = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002,0.005,0.01,0.02,0.1]
def bucket(val):
    for pos in xrange(len(buckets)-1):
        if buckets[pos+1]>val:
            return pos
    return len(buckets) -1

probabilities = np.zeros((data.num_tokens, num_buckets), dtype=int)
for [user_id, token_id, count, token_user_prob] in data.user_token_counts:
    probabilities[token_id, bucket(token_user_prob)] += 1


var_in_probs = probabilities.var(1)
best_tokens = {}
for i in xrange(data.num_tokens):
    best_tokens[data.tokens[i]] = var_in_probs[i]  * (1- data.percent_cover(data.tokens[i]))

number_tokens_wanted = 20000
sorted_tokens = sorted(best_tokens, key=best_tokens.get, reverse=True)
included_tokens = sorted_tokens[0:number_tokens_wanted]

def token_filter_by_variance(token):
    return (token in included_tokens)

data.write_to_csv('inc_', token_filter_fun=token_filter_by_variance)


# the rest is just debug - but it'd be nice to write this info to a file
np.set_printoptions(threshold='nan')
max_tokens = number_tokens_wanted
i=0
for token in included_tokens:
    print probabilities[data.token_ids[token]], token, best_tokens[token], data.percent_cover(token)
    i += 1
    if (i>max_tokens):
        break
