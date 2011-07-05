#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, os
import json
import gzip
from math import log
from collections import defaultdict
from common import open_maybe_gzip

TRIGRAM_COUNT_OFFSET = 0.5

#drink the hose example:
#{"friend_count": 1897, "statuses_count": 54759, "text": "ABC releases trailers for horror series The River and the dark fairy tale Once Upon A Time [Video]: \n\t\t\t\t\t\t\t\t\t\t\n... http://bit.ly/j2U0Ek", "profile_image_url": "http://a2.twimg.com/profile_images/1147470332/top_notched_logo_normal.JPG", "timezone": "Chennai", "geo": null, "id": 70579336008302592, "lang": "en", "screen_name": "Top_Notched", "created_at": "2011-05-17 19:59:59", "entities": {"user_mentions": [], "hashtags": [], "urls": [{"url": "http://bit.ly/j2U0Ek", "indices": [116, 136], "expanded_url": null}]}, "followers_count": 2225, "location": "Udaipur"}
#
# The interesting bits:
#
#{"text": "ABC releases ....", "id": 70579336008302592, "lang": "en", "screen_name": "Top_Notched"}
#
# NB: not all rows have the id field.

def debug(*args):
    for x in args:
        print >> sys.stderr, x,
    print >> sys.stderr

import re

#ignore numbers, perhaps with dollar signs, or decimal points
_number_re = re.compile(r'^-?\$?\d+\.?\d*$')
_split_re = re.compile(r'[!@#$%^&*_+|}{;":?><,./ ]+')

def update_lut_lowercase(lut, s):
    """Fill a trigram Look Up Table with utf-8 byte trigrams of
    lowercase text with normalised whitespace.

    'I am Pat!' => ' i ', 'i a', ' am', 'am ', 'm p', ' pa', 'pat', 'at!', 't! '

    #hash and @at tags and URLs and numbers are ignored.
    """
    if isinstance(s, unicode):
        s = s.encode('utf-8')

    s = ' '.join(x for x in s.lower().split() if not x[:4] == 'http' and not x[0] in '#@'
                 and not _number_re.match(x))
    size = len(s)
    if size != 0:
        s = ' ' + s + ' '
        for i in range(size):
            k = s[i : i + 3]
            lut[k] += 1
    return size

def update_lut_lowercase_depunctuated(lut, s):
    """Like update_lut_lowercase, but with most punctuation removed.

    'I am Pat!' => ' i ', 'i a', ' am', 'am ', 'm p', ' pa', 'pat', 'at '
    """
    if isinstance(s, unicode):
        s = s.encode('utf-8')

    s = ' '.join(x for x in s.lower().split() if not x[:4] == 'http' and not x[0] in '#@'
                 and not _number_re.match(x))
    s = ' '.join(x for x in _split_re.split(s) if x)
    size = len(s)
    if size != 0:
        s = ' ' + s + ' '
        for i in range(size):
            k = s[i : i + 3]
            lut[k] += 1
    return size


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
    (http://borel.slu.edu/crubadan/), and the trigram files are
    interchangeable with its ones (though they are GPLv3).

    #hash and @at tags and URLs and numbers are ignored.
    """
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    #clear out urls before splitting on punctuation, otherwise url tails remain
    s = ' '.join(x for x in s.split() if not x[:4] == 'http' and not x[0] in '#@'
                 and not _number_re.match(x))
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

def update_lut_word_aware_lc(lut, s):
    """Like update_lut_word_aware, but putting everything in lowercase.

    'I am Pat!' => '<i>', '<am', 'am>', '<pa', 'pat', 'at>'
    """
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    #clear out urls before splitting on punctuation, otherwise url tails remain
    s = ' '.join(x for x in s.lower().split() if not x[:4] == 'http' and not x[0] in '#@'
                 and not _number_re.match(x))
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

class Trigram:
    """From one or more text files, the frequency of three character
    sequences is calculated.  When treated as a vector, this information
    can be compared to other trigrams, and the difference between them
    seen as an angle.  The cosine of this angle varies between 1 for
    complete similarity, and 0 for utter difference.  This is used to
    determine the language of a body of text.
    """
    trigrams = 0
    log_evidence = None
    def __init__(self, fn=None, mode='word_aware'):
        #XXX python2.7 and 3.1 have Counter(), which is an advancement on defaultdict(int)
        self.lut = defaultdict(int)
        self.mode = mode
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
        f.close()

    def calculate_entropy(self, count_offset=TRIGRAM_COUNT_OFFSET,
                          other=None, other_mix=0.5):
        """Create and store a table indicating how many bits of
        evidence each trigram contains for this hypothesis over the
        alternative.  Where the alternative is indicated, the table
        stores a negative number.

        The default alternative hypothesis is a uniform distribution
        with weights normalised as if it had the same number of
        observed trigrams as this hypothesis, but evenly spread out
        (this is arbitrary).

        If other is given, it should be another Trigram to use as the
        alternative model.  In that case, other_mix indicates a ratio
        between other and the default flat model.  If other_mix is 1,
        none of the default model is used.  If other_mix is 0, other
        is not used at all.  In between you get a linear blend.

        The count_offset parameter is added to all trigram count
        values.  A lower number treats unknown trigrams as less
        probable.  Zero is forbidden.
        """
        #the number of possible trigrams
        all_tgms = 256 * 256 * 256
        known_tgms = len(self.lut)
        unknown_tgms = all_tgms - known_tgms

        self.total_count = float(sum(self.lut.itervalues())) + all_tgms * count_offset
        self.min_evidence = log(count_offset, 2)
        self.uniform_evidence = log(self.total_count / all_tgms, 2)
        self.log_evidence = dict((k, log(v + count_offset, 2) - self.uniform_evidence)
                                  for k, v in self.lut.iteritems())


        if other is not None and other_mix != 0:
            if other.log_evidence is None:
                other.calculate_entropy(count_offset=count_offset)
            u = self.uniform_evidence * (1.0 - other_mix)
            for k, ov in other.log_evidence.iteritems():
                sv = self.log_evidence.get(k, self.uniform_evidence)
                v = sv + u - ov * other_mix
                self.log_evidence[k] = v
            self.uniform_evidence = u

        debug("min evidence is ", self.min_evidence,
              "evidence('the')", self.log_evidence['the'],
              "uniform evidence", self.uniform_evidence
              )


    def probable_similarity(self, other):
        """On average, how many bits if evidence are there per trigram
        for the English hypothesis vs the uniform random string
        hypothesis.
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
            total += bitlut.get(k, self.min_evidence - self.uniform_evidence) * v

        return total / len2

    def hose_filter(self, infile):
        """Read the drink_the_hose json lines in infile and yield
        similarity."""
        if hasattr(infile, 'next'):
            f = infile
        else:
            f = open_maybe_gzip(infile)
        for line in f:
            j = json.loads(line)
            s = j['text'].encode('utf8')
            p = self.probable_similarity(s)
            yield {'score': p,
                   'text': s,
                   #'id': j['id'],
                   'screen_name': j["screen_name"].encode('utf8')
                   }
        if f is not infile:
            f.close()

    def save_trigrams(self, filename):
        """Save the trigram data to a file.

        The format of the file is specific to the trigram mode.

        It is much quicker to load the trigrams from a trigram file
        than to regenerate the model from raw text."""
        values = [(v, k) for k, v in self.lut.iteritems()]
        values.sort()
        values.reverse()
        f = open_maybe_gzip(filename, 'w')
        for count, tg in values:
            print >> f, count, tg
        f.close()

    def load_trigrams(self, filename):
        """Load model data from a text file where each formatted thus:

        <number of occurances><space><the three bytes><end of line>

        The format of the trigram is specific to the trigram mode, but
        no attempt is made to ensure that it is right.
        """
        f = open_maybe_gzip(filename)
        for line in f:
            count, tg = line.rstrip('\n').split(' ', 1)
            self.lut[tg] += int(count)
        f.close()

def text_to_trigrams(text_name, trigram_name, mode):
    """save a text file in trigramised form"""
    tg = Trigram(mode=mode)
    tg.import_text(text_name)
    tg.save_trigrams(trigram_name)


MODES = tuple(x[11:] for x in globals() if x.startswith('update_lut_'))
