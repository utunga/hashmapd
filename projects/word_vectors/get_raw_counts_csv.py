from optparse import OptionParser
import os, sys, time
from collections import defaultdict
from operator import itemgetter
import csv
import codecs

def get_git_home():
    testpath = '.'
    while not '.git' in os.listdir(testpath) and not os.path.abspath(testpath) == '/':
        testpath = os.path.sep.join(('..', testpath))
    if not os.path.abspath(testpath) == '/':
        return os.path.abspath(testpath)
    else:
        raise ValueError, "Not in git repository"

HOME = get_git_home()
sys.path.append(HOME)

from hashmapd.csv_unicode_helpers import UnicodeWriter
from hashmapd.tokenize import TweetTokenizer

from hashmapd.common import find_git_root, debug
BASE_DIR = find_git_root()
DEFAULT_PATH = "projects/word_vectors/tweets/"
TWEETS_DIR = os.path.join(BASE_DIR, DEFAULT_PATH)
ALL_TOKEN_COUNTS_FILE = 'all_token_counts.csv'
ALL_USER_TOTALS_FILE = 'all_user_totals.csv'
ALL_USER_TOKEN_COUNTS_FILE = 'all_user_token_counts.csv'

def write_token_counts(infile, username, token_user_counts):
    for token, count in sorted(token_user_counts.iteritems(), key=itemgetter(1), reverse=True):
        infile.write("'%s','%s',%i\n"%(unicode(username),unicode(token), count))

def write_user_totals(user_totals):
    debug('writing ' + ALL_USER_TOTALS_FILE)
    user_totals_file = codecs.open(ALL_USER_TOTALS_FILE, encoding="utf-8",mode= "w")
    for username, count in sorted(user_totals.iteritems(), key=itemgetter(1), reverse=True):
        user_totals_file.write("'%s',%i\n"%(unicode(username), count))
    user_totals_file.close()

def write_token_totals(token_totals):
    debug('writing ' + ALL_TOKEN_COUNTS_FILE)
    token_counts_file = codecs.open(ALL_TOKEN_COUNTS_FILE, encoding="utf-8",mode= "w")
    for token, count in sorted(token_totals.iteritems(), key=itemgetter(1), reverse=True):
        token_counts_file.write("'%s',%i\n"%(unicode(token), count))
    token_counts_file.close()

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", help="input json file to parse (.gz or not)")
    parser.add_option("-a", "--all", help="tokenize all json files from the base dir",action="store_true", default=False)
    parser.add_option("-d", "--basedir", help="base dir to use, defaults to $git_root/" + DEFAULT_PATH)
    parser.add_option("-M","--max", help="set maximum number of rows to parse - for testing", type="int")

    (options, args) = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()

    tweets_dir = options.basedir if (options.basedir) else TWEETS_DIR
    tokenizer = TweetTokenizer(tweets_dir)

    #if (options.max):
    #    print options.max>1
    #exit()
    
    #counter = Counter()

    if (options.input):
        overall_counter = defaultdict(int)
        for (username, token) in tokenizer.get_tokens_from_file(options.input):
            overall_counter[token] += 1
            
    if (options.all):
        overall_token_counts = defaultdict(int)
        token_user_counts = defaultdict(int)
        user_totals = defaultdict(int)
        debug('opening ' + ALL_USER_TOKEN_COUNTS_FILE)
        user_token_file = codecs.open(ALL_USER_TOKEN_COUNTS_FILE, encoding="utf-8",mode= "w")
        last_username = None
        row=0
        for (username, token) in tokenizer.get_tokens_from_all_files():
            if ((last_username) and last_username!=username):
                write_token_counts(user_token_file, last_username, token_user_counts)
                token_user_counts = defaultdict(int) #reset counter
            last_username = username
            token_user_counts[token] += 1
            overall_token_counts[token] += 1
            user_totals[username] += 1
            row += 1
            if (options.max and row>options.max):
                break
        #don't forget to write out the counts for that last user
        write_token_counts(user_token_file, last_username, token_user_counts)
        debug("closing " + ALL_USER_TOKEN_COUNTS_FILE)
        user_token_file.close()

        write_token_totals(overall_token_counts)
        write_user_totals(user_totals)

if __name__ == '__main__':
    main()
