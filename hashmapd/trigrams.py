#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, os
import json
import gzip
from math import log
from collections import defaultdict

#drink the hose example:
#{"friend_count": 1897, "statuses_count": 54759, "text": "ABC releases trailers for horror series The River and the dark fairy tale Once Upon A Time [Video]: \n\t\t\t\t\t\t\t\t\t\t\n... http://bit.ly/j2U0Ek", "profile_image_url": "http://a2.twimg.com/profile_images/1147470332/top_notched_logo_normal.JPG", "timezone": "Chennai", "geo": null, "id": 70579336008302592, "lang": "en", "screen_name": "Top_Notched", "created_at": "2011-05-17 19:59:59", "entities": {"user_mentions": [], "hashtags": [], "urls": [{"url": "http://bit.ly/j2U0Ek", "indices": [116, 136], "expanded_url": null}]}, "followers_count": 2225, "location": "Udaipur"}

def debug(*args):
    for x in args:
        print >> sys.stderr, x,
    print >> sys.stderr

def open_maybe_gzip(filename, mode='rb'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def update_lut_lowercase(lut, s):
    """Fill a trigram Look Up Table with utf-8 byte trigrams of
    lowercase text with normalised whitespace.

    #hash and @at tags and URLs are ignored.
    """
    if isinstance(s, unicode):
        s = s.encode('utf-8')

    s = ' '.join(x for x in s.lower().split() if not x[:4] == 'http' and not x[0] in '#@'
                 and not x.isdigit())
    size = len(s)
    if size == 0:
        return 0
    s = ' ' + s + ' '
    for i in range(size):
        k = s[i : i + 3]
        lut[k] += 1
    return size


import re
_split_re = re.compile(r'[!@#$%^&*_+|}{;":?><,./ ]+')
def update_lut_word_aware(lut, s):
    """Fill a trigram Look Up Table with utf-8 byte trigrams that
    explicitly mark word boundaries.

    < marks the start of a word.
    > marks the end.

    < and > can only occur at the beginning and end of the trigram,
    respectively.  Spaces or other characters between words are not
    encoded, but case is retained.  For example:

    'I am Pat!' => '<I>', '<am', 'am>', '<Pa', 'Pat', 'at>'

    Note that the trigrams never look across the space between words,
    so the frequency of, say, 'I a' is ignored.

    Various punctuation characters are treated as word boundaries and
    discarded.

    This format is used in An Crúbadán by Kevin P. Scannell
    (http://borel.slu.edu/crubadan/).

    #hash and @at tags and URLs are ignored.
    """
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    #clear out urls before splitting on punctuation, otherwise url tails remain
    s = ' '.join(x for x in s.split() if not x[:4] == 'http' and not x[0] in '#@')
    #split on all non-words
    bits = _split_re.split(s)
    #print bits
    trigrams = 0
    for b in bits:
        if b:
            size = len(b)
            trigrams += size
            b = '<' + b + '>'
            for i in range(size):
                k = b[i : i + 3]
                lut[k] += 1
    return trigrams

def norm(lut):
    """Find the Euclidean length of a vector stored in a dictionary's
    values"""
    return sum(x * x for x in lut.itervalues()) ** 0.5


class Trigram:
    """From one or more text files, the frequency of three character
    sequences is calculated.  When treated as a vector, this information
    can be compared to other trigrams, and the difference between them
    seen as an angle.  The cosine of this angle varies between 1 for
    complete similarity, and 0 for utter difference.  This is used to
    determine the language of a body of text.
    """
    norm = 0
    trigrams = 0
    log_evidence = None
    def __init__(self, fn=None, mode='word_aware'):
        #XXX python2.7 and 3.1 have Counter(), which is an advancement on defaultdict(int)
        self.lut = defaultdict(int)
        if fn is not None:
            self.import_text(fn)

        self.update_lut = globals()['update_lut_' + mode]

    def import_text(self, fn):
        """Update the model with the character trigrams in the named
        file.
        """
        f = open_maybe_gzip(fn)
        s = f.read()
        #normalise as canonical utf8
        for encoding in ('utf8', 'iso-8859-1'):
            try:
                s = s.decode(encoding)
                break
            except UnicodeDecodeError:
                pass
        s = s.encode('utf8')

        f.close()
        self.trigrams += self.update_lut(self.lut, s)
        self.norm = norm(self.lut)

    def import_json(self, fn, by_line=True):
        """Import JSON or drink_the_hose style pseudo-JSON.
        Drink_the_hose stores each tweet on its own line.
        Real JSON would need them to be in square brackets and separated by commas.

        @param by_line whether to use drink_the_hose style (True by default)
        """
        f = open_maybe_gzip(fn)
        if by_line:
            tweets = (json.loads(line)['text'] for line in f)
        else:
            tweets = (x['text'] for x in json.load(f))

        for s in tweets:
            s = s.encode('utf8')
            self.trigrams += self.update_lut(self.lut, s)
        self.norm = norm(self.lut)
        f.close()

    def cosine_similarity(self, other, adaptive=False):
        """returns a number between 0 and 1 indicating similarity
        between this model and another trigram or string.

        1 means an identical ratio of trigrams;
        0 means no trigrams in common.
        """
        if isinstance(other,  str):
            lut2 = defaultdict(int)
            trigrams2 = self.update_lut(lut2, other)
            norm2 = norm(lut2)

        elif isinstance(other, Trigram):
            lut2 = other.lut
            norm2 = other.norm
            trigrams2 = other.trigrams

        else:
            raise TypeError("Can't compare Trigram with %s" % type(other))

        if norm2 == 0:
            return 0

        lut1 = self.lut
        # Iterate over smallest dictionary (probably the "other" one,
        # but let's not presume).
        if len(lut1) < len(lut2):
            lut1, lut2 = lut2, lut1

        total = 0
        for k in lut2.keys():
            if k in lut1:
                total += lut1[k] * lut2[k]

        return float(total) / (self.norm * norm2)

    def calculate_entropy(self):
        all_tgms = 256 * 256 * 256
        known_tgms = len(self.lut)
        unknown_tgms = all_tgms - known_tgms

        # all trigram values have 1 added to prevent log(0)
        self.log_evidence = dict((k, log(v + 1, 2)) for k, v in self.lut.iteritems())
        self.min_evidence = log(1, 2)

        total_count = sum(self.lut.itervalues()) + all_tgms

        self.uniform_evidence = log(float(total_count) / all_tgms, 2)

    def probable_similarity(self, other, adaptive=False):
        """1 means an identical ratio of trigrams;
        0 means no trigrams in common.
        """
        if self.log_evidence is None:
            self.calculate_entropy()
        lut2 = defaultdict(int)
        len2 = self.update_lut(lut2, other)
        if len2 == 0: #no evidence
            return 0

        bitlut = self.log_evidence
        total = 0.0
        for k, v in lut2.iteritems():
            total += bitlut.get(k, self.min_evidence) * v

        log_odds = total - len2 * self.uniform_evidence
        return log_odds / len2

    def __sub__(self, other):
        """indicates difference between trigram sets; 1 is entirely
        different, 0 is entirely the same."""
        return 1 - self.cosine_similarity(other)

    def filter_the_hose(self, infile, outfile, rejectfile=None, threshold=0.9, adaptive=False):
        """Read the drink_the_hose json lines in infile and spit then
        out to outfile if they meet the threshold of similarity to
        this model's corpus."""
        f = open_maybe_gzip(infile)
        fout = open_maybe_gzip(outfile, 'w')
        if rejectfile is not None:
            frej = open_maybe_gzip(rejectfile, 'w')
        _threshold = threshold
        for line in f:
            s = json.loads(line)['text'].encode('utf8')
            #p = self.cosine_similarity(s)
            p = self.probable_similarity(s)
            #log(s, p)
            if adaptive:
                _threshold = threshold * min(1, (len(s) + 1) / 120.0)

            if p >= _threshold:
                #fout.write(line)
                fout.write("%5f %s\n" % (p, s))
            elif rejectfile is not None:
                #frej.write(line)
                frej.write("%5f %s\n" % (p, s))

        f.close()
        fout.close()

    def save_trigrams(self, filename):
        values = [(v, k) for k, v in self.lut.iteritems()]
        values.sort()
        values.reverse()
        f = open_maybe_gzip(filename, 'w')
        for count, tg in values:
            print >> f, count, tg
        f.close()

    def load_trigrams(self, filename):
        f = open_maybe_gzip(filename)
        for line in f:
            count, tg = line.rstrip('\n').split(' ', 1)
            self.lut[tg] += int(count)
        f.close()
        self.norm = norm(self.lut)

def test():
    try:
        mode = sys.argv[1]
    except IndexError:
        mode = 'word_aware'
    from subprocess import Popen, PIPE
    p = Popen(["git", "rev-parse", "--show-toplevel"], stdout=PIPE)
    root = p.communicate()[0].strip()
    if p.returncode:
        "Can't find the git tree"
        sys.exit(1)
    import time
    t = time.time()
    #http://ia600402.us.archive.org/5/items/myTweets/17662317-A-Thousand-Tweets_merged_djvu.txt
    tg = Trigram(mode=mode)
    if 0:
        for fn in (
            "corpi/raw/17662317-A-Thousand-Tweets_merged_djvu.txt",
            "corpi/raw/1933-Roosevelt.txt",
            "corpi/raw/1961-Kennedy.txt",
            "corpi/raw/2009-Obama.txt",
            "corpi/raw/carroll-alice.txt",
            "corpi/raw/dasher_training_english_GB.txt",
            "corpi/raw/english-web.txt",
            "corpi/raw/lulz.txt",
            "corpi/raw/enron-sent.gz",
            "corpi/raw/wikipedia.txt",
            "corpi/raw/irc.txt.gz",
            "corpi/raw/bash-org.txt",
            ):
            fn = os.path.join(root, fn)
            if True:
                # save the corpus in trigramised form
                tg2 = Trigram(mode=mode)
                tg2.import_text(fn)
                tg2.save_trigrams(fn.replace('raw/', 'trigram/')
                                  .replace('.txt', '-%s.txt' % mode)
                                  .replace('.gz', ''))

            tg.import_text(fn)
            t2 = time.time()
            debug("got %s at %s" % (fn, t2 - t))
        if 0 and mode == 'word_aware':
            for fn in (
                "corpi/trigram/en-3grams.txt",
                ):
                tg.load_trigrams(fn)
                t2 = time.time()
                debug("got %s at %s" % (fn, t2 - t))
        tg.save_trigrams(os.path.join(root, 'corpi/trigram/trigrams-%s.txt' % mode))
    else:
        tg.load_trigrams(os.path.join(root, 'corpi/trigram/trigrams-%s.txt' % mode))
        t2 = time.time()
        debug("got %s trigrams at %s" % (mode, t2 -t))
    threshold = {
        'lowercase':  10.5,
        'word_aware': 10.5,
        }[mode]
    tg.filter_the_hose(os.path.join(root, "stash/drink-the-hose-2011051103.txt.gz"),
                       "/tmp/%s-good.txt" % mode,
                       "/tmp/%s-rejects.txt" % mode,
                       threshold=threshold, adaptive=False)

    t2 = time.time()
    debug("filtered hose at %s" % (t2 - t,))



if __name__ == '__main__':
    test()
