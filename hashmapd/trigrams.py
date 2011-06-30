#!/usr/bin/python
import sys
import json
import gzip
from collections import defaultdict

#drink the hose example:
#{"friend_count": 1897, "statuses_count": 54759, "text": "ABC releases trailers for horror series The River and the dark fairy tale Once Upon A Time [Video]: \n\t\t\t\t\t\t\t\t\t\t\n... http://bit.ly/j2U0Ek", "profile_image_url": "http://a2.twimg.com/profile_images/1147470332/top_notched_logo_normal.JPG", "timezone": "Chennai", "geo": null, "id": 70579336008302592, "lang": "en", "screen_name": "Top_Notched", "created_at": "2011-05-17 19:59:59", "entities": {"user_mentions": [], "hashtags": [], "urls": [{"url": "http://bit.ly/j2U0Ek", "indices": [116, 136], "expanded_url": null}]}, "followers_count": 2225, "location": "Udaipur"}


def log(*args):
    for x in args:
        print >> sys.stderr, x,
    print >> sys.stderr

def open_maybe_gzip(filename, mode='rb'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def update_lut(lut, s):
    if isinstance(s, unicode):
        s = s.encode('utf-8')

    s = ' '.join(x for x in s.lower().split() if not x[:4] == 'http' and not x[0] in '#@')
    for i in range(len(s) - 2):
        k = s[i : i + 2]
        lut[k] += 1
    length = sum(x * x for x in lut.itervalues()) ** 0.5
    return length


class Trigram:
    """From one or more text files, the frequency of three character
    sequences is calculated.  When treated as a vector, this information
    can be compared to other trigrams, and the difference between them
    seen as an angle.  The cosine of this angle varies between 1 for
    complete similarity, and 0 for utter difference.  This is used to
    determine the language of a body of text.
    """
    length = 0

    def __init__(self, fn=None):
        #XXX python2.7 and 3.1 have Counter(), which is an advancement on defaultdict(int)
        self.lut = defaultdict(int)
        if fn is not None:
            self.import_text(fn)

    def import_text(self, fn):
        f = open(fn)
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
        self.length = update_lut(self.lut, s)


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
            self.length = update_lut(self.lut, s)
        f.close()

    def similarity(self, other):
        """returns a number between 0 and 1 indicating similarity
        between this model and another trigram or string.

        1 means an identical ratio of trigrams;
        0 means no trigrams in common.
        """
        if isinstance(other,  str):
            lut2 = defaultdict(int)
            length2 = update_lut(lut2, other)

        elif isinstance(other, Trigram):
            lut2 = other.lut
            length2 = other.length

        else:
            raise TypeError("Can't compare Trigram with %s" % type(other))

        if length2 == 0:
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

        return float(total) / (self.length * length2)

    def __sub__(self, other):
        """indicates difference between trigram sets; 1 is entirely
        different, 0 is entirely the same."""
        return 1 - self.similarity(other)


    def filter_the_hose(self, infile, outfile, rejectfile=None, threshold=0.9):
        """Read the drink_the_hose json lines in infile and spit then
        out to outfile if they meet the threshold of similarity to
        this model's corpus."""
        f = open_maybe_gzip(infile)
        fout = open_maybe_gzip(outfile, 'w')
        if rejectfile is not None:
            frej = open_maybe_gzip(rejectfile, 'w')
        for line in f:
            s = json.loads(line)['text'].encode('utf8')
            p = self.similarity(s)
            #log(s, p)
            if p >= threshold:
                #fout.write(line)
                fout.write("%5f %s\n" % (p, s))
            elif rejectfile is not None:
                #frej.write(line)
                frej.write("%5f %s\n" % (p, s))

        f.close()
        fout.close()


def test():
    #http://ia600402.us.archive.org/5/items/myTweets/17662317-A-Thousand-Tweets_merged_djvu.txt
    tg = Trigram()
    for fn in (
        "stash/17662317-A-Thousand-Tweets_merged_djvu.txt",
        "stash/1933-Roosevelt.txt",
        "stash/1961-Kennedy.txt",
        "stash/2009-Obama.txt",
        "stash/carroll-alice.txt",
        "stash/dasher_training_english_GB.txt",
        #"stash/en-3grams.txt",
        "stash/english-web.txt",
        "stash/lulz.txt",
        ):
        tg.import_text(fn)
    tg.filter_the_hose("stash/drink-the-hose-2011051103.txt.gz", "/tmp/good.txt", "/tmp/rejects.txt",
                       threshold=0.5)



if __name__ == '__main__':
    test()
