import sys, os
import json
import re
from urlparse import urlparse
from common import open_maybe_gzip
from trigrams import debug

#ignore numbers, perhaps with dollar signs, or decimal points
_number_re = re.compile(r'^-?\$?\d+\.?\d*$')
_split_re = re.compile(r'[!@#$%^&*_+|}{;":?><,./ ]+')

#match 4 or more repetitions of one or two characters ("hahahaha", "okkkkkkaaay")
_repeat_re = re.compile(r'(..??)\1\1\1+')

#simple re for splitting - anything *not* a word is a seperator
_simple_wordsplitter_re = re.compile(r'[^\w]+')

def get_domain(s):
    hostname = urlparse(s).hostname
    if (hostname):
        return "http://"+ ".".join(part for part in hostname.lower().split(".") if not part=='www')
    return None

def get_domains(s):
    hostname = urlparse(s).hostname
    if (hostname):
        parts = [part for part in hostname.lower().split(".") if not part=="www"]
        return ["http://"+ ".".join(parts[-i:]) for i in xrange(len(parts)+1) if i >1]
    # get_domains("http://www.foo.bar.com/asdf
    # gives ['http://bar.com', 'http://foo.bar.com']

def strip_sanitize_lc(s):
    s = s.replace('&lt;', '<').replace('&gt;', '>')
    s = s.replace('&amp;', '&')
    s = s.replace('\'', '')
    s = s.replace('"', '')
    s = _repeat_re.sub(r"\1\1\1", s)

    #clear out urls, #hashtags and @mentions
    s = ' '.join(x for x in s.lower().split()
                 if not x[:4] == 'http'
                 and not x[0] in '#@')
                 #and not x == 'rt'            #not 'RT' because s.lower()
                 #and not _number_re.match(x))
    return s

class TweetTokenizer:
    """Tokenize twitter json into words used aka tokens (with some special handling for urls @replies and #hashtags)
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_tokens_from_all_files(self):
        print 'Will walk files in ' + self.base_dir
        for (dir,drop,files) in list(os.walk(self.base_dir)):
            print 'Entering ' + dir
            for file in files:
                filename = os.path.join(dir, file)
                print 'Tokenizing ' + filename
                for tuple in self.get_tokens_from_file(filename):
                    yield tuple

    def get_tokens_from_file(self, filename):
        """Read the json lines from specified file and yield tokens."""

        f = open_maybe_gzip(filename)
        for line in f:
            if isinstance(line, unicode):
                line = line.encode('utf-8')
            j = json.loads(line)
            doc_type = j['doc_type']

            if doc_type == 'raw_tweet':
                screen_name = "@" + j['username']
                for token in self.get_tokens_from_raw_tweet(j):
                    yield (screen_name.lower(), token)

            elif doc_type == 'csharp_munged_tweet':
                screen_name = "@" + j['screen_name']
                for token in self.get_tokens_from_csharp_munged_tweet(j):
                    yield (screen_name.lower(), token)
                    
            else:
                raise Exception("unexpected doc type ", doc_type)
        f.close()

    def get_tokens_from_raw_tweet(self, json):
        raw = json['text'].encode('utf8')
        entities = json['entities']
        #debug(raw)
        
        # get entities mentioned
        for mention in entities['user_mentions']:
            yield '@'+mention['screen_name'].lower()
        for hashtag in entities['hashtags']:
            yield '#' + hashtag['text'].lower()
        for url in entities['urls']:
            short_url = url['url']
            expanded_url = url['expanded_url']
            # return both expanded and short url form
            # as well as 'just the domain' for each
            yield short_url
            if (expanded_url):
                yield expanded_url
                for part_domain in get_domains(expanded_url):
                    yield part_domain
                yield get_domain(short_url)
            else:
                for part_domain in get_domains(short_url):
                    yield part_domain
        # get other words
        text = strip_sanitize_lc(raw)
        for token in _simple_wordsplitter_re.split(text):
            if (token): yield token

    def get_tokens_from_csharp_munged_tweet(self, json):

        raw = json['text']
        entities = json['entities']
        #debug(raw)

        #yield entities
        if (entities):
            for entity in entities:
                yield entity.lower()

        #strip out entities then yield the 'normal words'
        text = strip_sanitize_lc(raw)
        for token in _simple_wordsplitter_re.split(text):
            if (token): yield token
