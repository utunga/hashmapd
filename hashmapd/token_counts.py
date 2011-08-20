import sys, os

import codecs
from collections import defaultdict
from operator import itemgetter
from hashmapd.csv_unicode_helpers import UnicodeReader
from hashmapd.common import debug
import numpy as np
from common import find_git_root

DEFAULT_DATA_DIR = "projects/word_vectors/tweets/"
DEFAULT_PREFIX = 'all_'

class TokenCounts:
    """Handles primarily read-only access to token counts data
    """

    def __init__(self, file_prefix=None, data_dir=None):
        self.init_data()
        self.set_data_dir(data_dir if (data_dir) else DEFAULT_DATA_DIR)
        self.set_prefix(file_prefix if (file_prefix) else DEFAULT_PREFIX)

    def init_data(self):
        self.tokens = []
        self.users = []
        self.user_token_counts = np.array([]) #bulk of 'data' goes here
        self.distinct_users_per_token = np.array([])
        self.user_totals = {}
        self.token_totals = {}
        self.user_ids = {}
        self.token_ids = {}
        self.num_users = 0
        self.num_tokens = 0

    def set_data_dir(self,data_dir=None):
        if (data_dir):
            self.data_dir = os.path.join(find_git_root(),data_dir)

    def set_prefix(self, file_prefix=None):
        if (file_prefix):
            self.file_prefix = file_prefix
            self.user_token_counts_file = os.path.join(self.data_dir, self.file_prefix + 'user_token_counts.csv' )
            self.token_totals_file = os.path.join(self.data_dir, self.file_prefix + 'token_counts.csv' )
            self.user_totals_file = os.path.join(self.data_dir, self.file_prefix + 'user_totals.csv' )

    def load_from_csv(self, file_prefix=None, data_dir=None, min_token_count=None, max_token_count=None, min_user_total=None):
    #fixme need to support filter on load also 
        self.init_data()
        self.set_data_dir(data_dir)
        self.set_prefix(file_prefix)

        debug("loading user totals from %s.."%(self.user_totals_file))
        reader = UnicodeReader(open(self.user_totals_file), quotechar="'")
        users = []
        user_ids = {}
        user_totals = {}
        idx = 0
        for [user, countstr] in reader:
            count = int(countstr)
            if ((min_user_total and count<min_user_total)):
                #debug("skipping %s"%(user))
                continue
            user_totals[user] = count
            if (not user_ids.has_key(user)):
                users.append(user)
                user_ids[user] = len(users)-1
            idx += 1
            if (idx % 10000==0):
                debug('.. %i users'%idx)

        debug('.. %i users'%idx)
        debug('..done')
        self.users = users
        self.user_ids = user_ids
        self.user_totals = user_totals
        self.num_users = len(users)

        debug("loading token totals from %s.."%(self.token_totals_file))
        reader = UnicodeReader(open(self.token_totals_file), quotechar="'")
        tokens = []
        token_ids = {}
        token_totals = {}
        idx = 0
        for row in reader:
            if (len(row) == 2):
                [token, countstr] = row
                count = int(countstr)
                if ((min_token_count and count<min_token_count) or
                (max_token_count and count>max_token_count)):
                    #debug("skipping '%s': count not within range"%(token))
                    continue

                token_totals[token] = count
                if (not token_ids.has_key(token)):
                    tokens.append(token)
                    token_ids[token] = len(tokens)-1
                idx += 1
                if (idx % 10000==0):
                    debug('.. %i tokens'%idx)

            else:
                debug("skipping '%s' incorrect row format"%(token))
                #raise Exception('unexpected input:',row)
        debug('.. %i tokens'%idx)
        debug('..done')
        self.tokens = tokens
        self.token_ids = token_ids
        self.token_totals = token_totals
        self.num_tokens = len(tokens)


        #load user_token_counts.csv


        debug("loading user token counts from %s.."%(self.user_token_counts_file))
        reader = UnicodeReader(open(self.user_token_counts_file), quotechar="'")
        idx = 0
        distinct_users_per_token = defaultdict(int)
        user_token_counts = []
        for row in reader:
            if (len(row)==3):
                [user, token, countstr] = row
                count = int(countstr)
                if (token_ids.has_key(token) and user_ids.has_key(user)):
                    token_user_prob = float(count) / self.user_totals[user]
                    user_token_counts.append((user_ids[user], token_ids[token], count, token_user_prob))
                    distinct_users_per_token[token] += 1
                    idx += 1
                    if (idx % 100000==0):
                        debug('.. %i user token counts'%idx)
            #else:
                #just skip this row, for now

        #want to do this, but.. it ruins the numpy operation happiness
        #        self.user_token_counts = np.array(user_token_counts,
        #                        dtype=np.dtype([('user_id', np.int32),
        #                                        ('token_id', np.int32),
        #                                        ('count', np.int32),
        #                                        ('prob', np.float32)]))

        self.user_token_counts = np.array(user_token_counts, dtype=np.float32)
        self.distinct_users_per_token = distinct_users_per_token
        debug('.. %i user token counts'%idx)
        debug('..done')

    def pass_anything_fun(self, user_or_token):
        return True

    def percent_cover(self, token):
        return float(self.distinct_users_per_token[token]) / self.num_users

    def write_to_csv(self, file_prefix=None, data_dir=None, user_filter_fun=None, token_filter_fun=None):
        self.set_data_dir(data_dir)
        self.set_prefix(file_prefix)
        user_filter_fun = user_filter_fun if (user_filter_fun) else self.pass_anything_fun
        token_filter_fun = token_filter_fun if (token_filter_fun) else self.pass_anything_fun

        debug('writing  %s ..'%(self.user_token_counts_file))
        included_user_totals = defaultdict(int)
        included_token_totals = defaultdict(int)
        idx =0
        outfile = codecs.open(self.user_token_counts_file, encoding="utf-8",mode= "w")
        for [user_id, token_id, count, prob] in self.user_token_counts:
            user = self.users[int(user_id)]
            token = self.tokens[int(token_id)]
            if (user_filter_fun(user) and token_filter_fun(token)):
                outfile.write("'%s','%s',%i\n"%(unicode(user),unicode(token), count))
                included_user_totals[user] += count
                included_token_totals[token] += count
                idx += 1
                if (idx % 100000==0):
                    debug('.. %i rows'%idx)

        outfile.close()
        debug("done")

        debug('writing %s ..'%(self.user_totals_file))
        user_totals_file = codecs.open(self.user_totals_file, encoding="utf-8",mode= "w")
        for username, count in sorted(included_user_totals.iteritems(), key=itemgetter(1), reverse=True):
            user_totals_file.write("'%s',%i\n"%(unicode(username), count))
        user_totals_file.close()
        debug("done")

        debug('writing %s ..'%(self.token_totals_file))
        token_totals_file = codecs.open(self.token_totals_file, encoding="utf-8",mode= "w")
        for token, count in sorted(included_token_totals.iteritems(), key=itemgetter(1), reverse=True):
            token_totals_file.write("'%s',%i\n"%(unicode(token), count))
        token_totals_file.close()
        debug("done")