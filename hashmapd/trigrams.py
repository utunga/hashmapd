#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, os
import json
import gzip
from math import log
from collections import defaultdict
from common import open_maybe_gzip

#Multiplied by mean trigram weight and added to each, to give unseen trigrams a chance
TRIGRAM_OFFSET_FACTOR = 0.5
ANTI_TRIGRAM_OFFSET_FACTOR = 1.0

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

#match 4 or more repetitions of one or two characters ("hahahaha", "okkkkkkaaay")
_repeat_re = re.compile(r'(..??)\1\1\1+')

def sanitise_string(s):
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    #clear out urls before splitting on punctuation, otherwise url tails remain
    return ' '.join(x for x in s.split() if not x[:4] == 'http'
                    and not x[0] in '#@'
                    and not x == 'RT'
                    and not _number_re.match(x))

def sanitise_string_lc(s):
    if isinstance(s, unicode):
        s = s.encode('utf-8')
    #clear out urls before splitting on punctuation, otherwise url tails remain
    s = ' '.join(x for x in s.lower().split() if not x[:4] == 'http'
                 and not x[0] in '#@'
                 and not x == 'rt'            #not 'RT' because s.lower()
                 and not _number_re.match(x))
    s = s.replace('&lt;', '<').replace('&gt;', '>')
    s = _repeat_re.sub(r"\1\1\1", s)
    return s


def update_lut_lowercase(lut, s):
    """Fill a trigram Look Up Table with utf-8 byte trigrams of
    lowercase text with normalised whitespace.

    'I am Pat!' => ' i ', 'i a', ' am', 'am ', 'm p', ' pa', 'pat', 'at!', 't! '

    #hash and @at tags and URLs and numbers are ignored.
    """
    s = sanitise_string_lc(s)
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
    s = sanitise_string_lc(s)
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
    s = sanitise_string(s)
    #split on all non-words
    bits = _split_re.split(s)
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
    s = sanitise_string_lc(s)
    bits = _split_re.split(s)
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
    """Model a language based on the frequency of different three
    character sequences.
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

    def calculate_entropy(self, offset_factor=TRIGRAM_OFFSET_FACTOR,
                          other=None, other_offset_factor=ANTI_TRIGRAM_OFFSET_FACTOR):
        """Create and store a table indicating how many bits of
        evidence each trigram contains for this hypothesis over the
        alternative.  Where the alternative is indicated, the table
        stores a negative number.

        The default alternative hypothesis is a uniform distribution
        with weights normalised as if it had the same number of
        observed trigrams as this hypothesis scaled by
        other_offset_factor, and evenly spread out.

        The offset_factor parameter is added to all trigram count
        values.  A lower number treats unknown trigrams as less
        probable.  Zero is forbidden, unless all 16M possible trigrams
        have been counted at least once.  The higher this number is,
        the less surprising an unseen trigram is to the model.  It is
        the background noise.

        If other is given, it should be another Trigram to use as the
        alternative model.  If that Trigram hasn't had
        calculate_entropy called, that is done with the offset_factor
        set to other_offset_factor.  Typically, other_offset_factor
        would be higher than offset_factor, which would make random
        trigrams more attractive to the other Trigram.
        """
        #the number of possible trigrams
        all_tgms = 256 * 256 * 256
        known_tgms = len(self.lut)
        unknown_tgms = all_tgms - known_tgms

        total_count = float(sum(self.lut.itervalues()))
        mean_count = total_count / all_tgms
        count_offset = mean_count * offset_factor
        adj_count = total_count + all_tgms * count_offset
        adj_mean = mean_count + count_offset
        log_norm = log(1.0 / adj_count, 2)
        self.min_evidence = log(count_offset, 2) + log_norm

        if other is None:
            # The opposing hypothesis is uniform noise
            if other_offset_factor == 0:
                #no opposition
                noise = 0
            else:
                noise = log(adj_mean * other_offset_factor, 2) + log_norm
            debug("noise is ", noise)
            self.default_evidence = self.min_evidence - noise
            self.log_evidence = dict((k, log(v + count_offset, 2) - noise + log_norm)
                                     for k, v in self.lut.iteritems())
        else:
            #if the other one already has its entropy set up, we use that
            #and other_offset_factor is ignored.
            if other.log_evidence is None:
                other.calculate_entropy(offset_factor=other_offset_factor,
                                        other_offset_factor=0)

            #the unopposed evidence for this hypothesis
            log_evidence = dict((k, log(v + count_offset, 2) + log_norm)
                                for k, v in self.lut.iteritems())

            self.default_evidence = self.min_evidence - other.min_evidence
            self.log_evidence = {}
            for k in set(other.log_evidence.keys()) | set(log_evidence.keys()):
                ov = other.log_evidence.get(k, other.min_evidence)
                sv = log_evidence.get(k, self.min_evidence)
                self.log_evidence[k] = sv - ov


        debug("log_norm is ", log_norm, "len is ", len(self.lut))
        debug("total count is ", total_count)
        debug("adjusted count is ", adj_count, "average is", adj_count / all_tgms)
        debug("min evidence is ", self.min_evidence,)
        debug("default evidence is ", self.default_evidence,
              "evidence('the')", self.log_evidence['the'],
              "evidence('los')", self.log_evidence['los'],
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
        default = self.default_evidence
        total = 0.0
        for k, v in lut2.iteritems():
            total += bitlut.get(k, default) * v

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
