#!/usr/bin/python
import os, sys
from collections import defaultdict

from hashmapd.trigrams import Trigram, debug, text_to_trigrams, MODES

from hashmapd.common import find_git_root, open_maybe_gzip
BASE_DIR = find_git_root()

RAW_CORPI_DIR = os.path.join(BASE_DIR, "corpi/raw/")
COOKED_CORPI_DIR = os.path.join(BASE_DIR, "corpi/trigram/")
TRIGRAM_MODEL_DIR = os.path.join(BASE_DIR, "corpi/compiled/")

#this might not be so on other machines
STASH_DIR = os.path.join(BASE_DIR, "stash")

CORPI = (#name, gzipped, pre-trigrammised
    ("17662317-A-Thousand-Tweets_merged_djvu", None, False),
    ("1933-Roosevelt", None, False),
    ("1961-Kennedy", None, False),
    ("2009-Obama", None, False),
    ("carroll-alice", None, False),
    ("dasher_training_english_GB", None, False),
    ("english-web", None, False),
    ("lulz", None, False),
    ("enron-sent", "gz", False),
    ("wikipedia", None, False),
    ("irc", "gz", False),
    ("bash-org", None, False),
    )

ANTI_CORPI = (
    ("anti-english", None, False),
)

def raw_corpi_path(base, gz):
    return os.path.join(RAW_CORPI_DIR,
                        base + ('.txt.gz' if gz else '.txt'))

def cooked_corpi_path(base, mode, gz=False):
    return os.path.join(COOKED_CORPI_DIR,
                        "%s-%s.%s" % (base, mode, ('txt.gz' if gz else 'txt')))


def pre_cook(modes=MODES, corpi=CORPI):
    if isinstance(modes, str):
        modes = [mode]
    for mode in modes:
        for base, gz, precooked in corpi:
            dest = cooked_corpi_path(base, mode)
            src = raw_corpi_path(base, gz)
            text_to_trigrams(src, dest, mode)


def load_corpi(mode, corpi=CORPI):
    if isinstance(mode, Trigram):
        tg = mode
        mode = tg.mode
    else:
        tg = Trigram(mode=mode)
    for base, gz, precooked in corpi:
        if precooked:
            tg.load_trigrams(cooked_corpi_path(base, mode, gz))
        else:
            fn = raw_corpi_path(base, gz)
            tg.import_text(fn)
    return tg


def bisect_the_hose(trigram, infile, goodfile, rejectfile, threshold):
    f = open_maybe_gzip(infile)
    fgood = open_maybe_gzip(goodfile, 'w')
    frej = open_maybe_gzip(rejectfile, 'w')

    hose_filter = trigram.hose_filter(f)

    for d in hose_filter:
        if d['score'] > threshold:
            fgood.write("%(score)5f %(text)s\n" % d)
        else:
            frej.write("%(score)5f %(text)s\n" % d)

    f.close()
    fgood.close()
    frej.close()

def group_by_user(trigram, infile, users=None):
    if users is None:
        users = {}
    for d in trigram.hose_filter(infile):
        users.setdefault(d['screen_name'], []).append(d['score'])
    return d


def get_trigram(mode, use_raw=False, corpi=CORPI, name_stem=''):
    if use_raw:
        tg = load_corpi(mode, corpi)
        tg.save_trigrams(os.path.join(TRIGRAM_MODEL_DIR, '%s%s.txt' % (name_stem, mode)))
    else:
        tg = Trigram(mode=mode)
        tg.load_trigrams(os.path.join(TRIGRAM_MODEL_DIR, '%s%s.txt' % (name_stem, mode)))
    return tg

def get_anti_trigram(mode, use_raw=False):
    tg = get_trigram(mode, use_raw=use_raw,
                     corpi=ANTI_CORPI, name_stem='anti-'
                     )
    return tg

def get_trigram_with_antimodel(mode, use_raw=False):
    tg = get_trigram(mode, use_raw=use_raw)
    atg = get_anti_trigram(tg.mode, use_raw=use_raw)
    tg.calculate_entropy(other=atg, other_mix=0.7)
    return tg

def iter_stash():
    for fn in os.listdir(STASH_DIR):
        if fn.startswith('drink-the-hose'):
            yield(os.path.join(STASH_DIR, fn))


def _filter_the_hose(tg):
    threshold = {
        'lowercase':  10.5,
        'word_aware': 10.5,
        'word_aware_lc': 10.5,
        }[tg.mode]

    bisect_the_hose(tg, os.path.join(STASH_DIR, "drink-the-hose-2011051103.txt.gz"),
                    "/tmp/%s-good.txt" % tg.mode,
                    "/tmp/%s-rejects.txt" % tg.mode,
                    threshold=threshold)



def _group_by_user(tg, verbose=False):
    users = {}
    for fn in iter_stash():
        print "doing %s" % fn
        group_by_user(tg, fn, users)
        print "%s users" % len(users)
        if verbose:
            _users_report(users)
    return users

def _users_report(users):
    counts = defaultdict(int)
    for v in users.itervalues():
        counts[len(v)] += 1
    for k in range(1, max(counts.keys()) + 1):
        print "%3d %s" % (k, counts.get(k, '.'))

def partition_users(users, outfile, rejfile, threshold):
    fout = open_maybe_gzip(outfile, 'w')
    frej = open_maybe_gzip(rejfile, 'w')
    for k, v in users.iteritems():
        if len(v) == 1:
            mean = v[0]
        else:
            mean = sum(v) / len(v)
        f = (fout if mean >= threshold else frej)
        #f.write("%4f %s - %s\n" % (mean, k, ' '.join("%4f" % x for x in v)))
        f.write("%4f %s\n" % (mean, k))
    fout.close()
    frej.close()

def _filter_with_antimodel(tg, use_raw=False, threshold=5.5):
    bisect_the_hose(tg, os.path.join(STASH_DIR, "drink-the-hose-2011051103.txt.gz"),
                    "/tmp/%s-good-anti.txt" % tg.mode,
                    "/tmp/%s-rejects-anti.txt" % tg.mode,
                    threshold=threshold)



def main(argv):
    try:
        mode = argv[1]
    except IndexError:
        mode = 'word_aware_lc'
    use_raw=False
    #for mode in MODES:
    #    print mode, use_raw
    #    tg = get_trigram_with_antimodel(mode, use_raw=use_raw)
    #    _filter_with_antimodel(tg, use_raw=use_raw, threshold=5.45)
    #_filter_the_hose(tg)
    tg = get_trigram_with_antimodel(mode, use_raw=use_raw)
    users = _group_by_user(tg)
    _users_report(users)
    partition_users(users, '/tmp/good_users', '/tmp/bad_users', 5.45)

if __name__ == '__main__':
    #pre_cook()
    main(sys.argv)
