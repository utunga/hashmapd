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

TEST_FILE_1 = os.path.join(STASH_DIR, "drink-the-hose-2011051103.txt.gz")
TEST_FILE_2 = os.path.join(STASH_DIR, "drink-the-hose-2011050811.txt.gz")

DEFAULT_MODE = 'word_aware_lc'
DEFAULT_THRESHOLD = 0.5



CORPI = (#name, gzipped, pre-trigrammised
    ("presidents", False, True),
    ("carroll-alice", False, True),
    ("dasher_training_english_GB", False, True),
    ("english-web", False, True),
    ("lulz", False, True),
    ("enron-sent", True, True),
    ("wikipedia", False, True),
    ("irc", True, True),
    ("bash-org", False, True),
    ("barely-english", False, False),
    )

ANTI_CORPI = (
    ("anti-english", False, True),
    ("near-english", False, False),
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
            dest = cooked_corpi_path(base, mode, gz)
            src = raw_corpi_path(base, gz)
            text_to_trigrams(src, dest, mode)

def pre_cook_full(modes=MODES, corpi=CORPI):
    for mode in modes:
        print mode
        tg = get_trigram_with_antimodel(mode, use_raw=True, corpi=corpi)

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

def order_the_hose(trigram, infile, outfile):
    f = open_maybe_gzip(infile)
    fout = open_maybe_gzip(outfile, 'w')

    hose_filter = trigram.hose_filter(f)
    rows = [(d['score'], d['text']) for d in hose_filter]
    rows.sort()
    for r in rows:
        fout.write("%5f %s\n" % r)

    f.close()
    fout.close()


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

def get_trigram_with_antimodel(mode, use_raw=False, corpi=CORPI):
    tg = get_trigram(mode, use_raw=use_raw, corpi=corpi)
    atg = get_anti_trigram(tg.mode, use_raw=use_raw)
    tg.calculate_entropy(other=atg)
    return tg

def iter_stash():
    for fn in os.listdir(STASH_DIR):
        if fn.startswith('drink-the-hose'):
            yield(os.path.join(STASH_DIR, fn))


def _filter_the_hose(tg, threshold=DEFAULT_THRESHOLD, suffix='', src=TEST_FILE_1):
    bisect_the_hose(tg, os.path.join(src),
                    "/tmp/%s-good%s.txt" % (tg.mode, suffix),
                    "/tmp/%s-rejects%s.txt" % (tg.mode, suffix),
                    threshold=threshold)


def _filter_all_modes(suffix=''):
    for mode in MODES:
        tg = get_trigram_with_antimodel(mode)
        _filter_the_hose(tg)


def _order_the_hose(tg, suffix='', src=TEST_FILE_1):
    order_the_hose(tg, src, "/tmp/%s-sorted%s.txt" % (tg.mode, suffix))

def _order_all_modes(modes=MODES, suffix=''):
    for mode in modes:
        tg = get_trigram_with_antimodel(mode)
        _order_the_hose(tg, suffix)


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



def main(argv):
    try:
        mode = argv[1]
    except IndexError:
        mode = DEFAULT_MODE
    tg = get_trigram_with_antimodel(mode)
    if 1:
        _filter_the_hose(tg)
    else:
        users = _group_by_user(tg)
        _users_report(users)
        partition_users(users, '/tmp/good_users', '/tmp/bad_users', DEFAULT_THRESHOLD)

if __name__ == '__main__':
    _modes = [DEFAULT_MODE]
    #pre_cook(modes=_modes)
    #pre_cook_full(modes=_modes)
    #main(sys.argv)
    #_filter_all_modes()
    _order_all_modes(modes=_modes)
