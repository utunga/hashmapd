from optparse import OptionParser
import os, sys, time
from collections import defaultdict
from operator import itemgetter
import csv
from hashmapd.csv_unicode_helpers import UnicodeWriter
from hashmapd.tokenize import TweetTokenizer

from hashmapd.common import find_git_root, debug
BASE_DIR = find_git_root()
DEFAULT_PATH = "projects/word_vectors/tweets/"
TWEETS_DIR = os.path.join(BASE_DIR, DEFAULT_PATH)
RAW_COUNTS_FILE = 'raw_counts.csv'

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", help="input json file to parse (.gz or not)")
    parser.add_option("-a", "--all", help="tokenize all json files from the base dir",action="store_true", default=False)
    parser.add_option("-d", "--basedir", help="base dir to use, defaults to $git_root/" + DEFAULT_PATH)

    (options, args) = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()

    tweets_dir = options.basedir if (options.basedir) else TWEETS_DIR
    tokenizer = TweetTokenizer(tweets_dir)

    #counter = Counter()
    counter = defaultdict(int)

    if (options.input):
        for (username, token) in tokenizer.get_tokens_from_file(options.input):
            counter[token] += 1
            
    if (options.all):
        for (username, token) in tokenizer.get_tokens_from_all_files():
            counter[token] += 1

    #for (token, count) in counter.most_common(200):
    #    print token, count

    print 'writing coordinates to csv'
    writer = UnicodeWriter(open(RAW_COUNTS_FILE, 'wb'))
    #for (token, count) in counter.most_common():
    for token, count in sorted(counter.iteritems(), key=itemgetter(1), reverse=True):
        writer.writerow([token, "%i"%count])
        
if __name__ == '__main__':
    main()
