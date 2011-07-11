#!/usr/bin/python
import os, sys, time
from collections import defaultdict
from optparse import OptionParser

from hashmapd.trigrams import Trigram, debug, text_to_trigrams, MODES
from hashmapd.trigrams import TRIGRAM_OFFSET_FACTOR, ANTI_TRIGRAM_OFFSET_FACTOR


from hashmapd.common import find_git_root, open_maybe_gzip
BASE_DIR = find_git_root()

RAW_CORPI_DIR = os.path.join(BASE_DIR, "corpora/raw/")
COOKED_CORPI_DIR = os.path.join(BASE_DIR, "corpora/trigram/")
TRIGRAM_MODEL_DIR = os.path.join(BASE_DIR, "corpora/compiled/")

#this might not be so on other machines
STASH_DIR = os.path.join(BASE_DIR, "stash")

#TEST_FILE_1 = os.path.join(STASH_DIR, "drink-the-hose-2011051203.txt.gz")
TEST_FILE_1 = os.path.join(STASH_DIR, "drink-the-hose-2011051103.txt.gz")
#TEST_FILE_1 = os.path.join(STASH_DIR, "drink-the-hose-2011050811.txt.gz")

DEFAULT_MODE = 'word_aware_lc'
#DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLD = "LOL"



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
    if goodfile is None:
        goodfile = os.devnull
    if rejectfile is None:
        rejectfile = os.devnull
    fgood = open_maybe_gzip(goodfile, 'w')
    frej = open_maybe_gzip(rejectfile, 'w')
    if isinstance(threshold, str):
        threshold = trigram.probable_similarity(threshold)
        debug("threshold is", threshold)

    hose_filter = trigram.hose_filter(f)

    for d in hose_filter:
        if d['score'] >= threshold:
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

def get_trigram_with_antimodel(mode, use_raw=False, corpi=CORPI,
                               tof=TRIGRAM_OFFSET_FACTOR, atof=ANTI_TRIGRAM_OFFSET_FACTOR):
    tg = get_trigram(mode, use_raw=use_raw, corpi=corpi)
    atg = get_anti_trigram(tg.mode, use_raw=use_raw)
    tg.calculate_entropy(other=atg,  offset_factor=tof,
                         other_offset_factor=atof)
    return tg

def iter_stash(d):
    if not os.path.isdir(d):
        yield d
        return
    for fn in os.listdir(d):
        if fn.startswith('drink-the-hose'):
            yield(os.path.join(d, fn))

def _group_by_user(tg, src=STASH_DIR):
    users = {}
    for fn in iter_stash(src):
        print "doing %s" % fn
        group_by_user(tg, fn, users)
        print "%s users" % len(users)
    return users

def users_report(users):
    counts = defaultdict(int)
    for v in users.itervalues():
        counts[len(v)] += 1
    for k in range(1, max(counts.keys()) + 1):
        print "%3d %s" % (k, counts.get(k, '.'))

def partition_users(users, outfile, rejfile, threshold):
    if outfile is None:
        outfile = os.devnull
    if rejfile is None:
        rejfile = os.devnull
    fout = open_maybe_gzip(outfile, 'w')
    frej = open_maybe_gzip(rejfile, 'w')
    for k, v in users.iteritems():
        if len(v) == 1:
            mean = v[0]
        else:
            mean = sum(v) / len(v)
        f = (fout if mean >= threshold else frej)
        f.write("%4f %s\n" % (mean, k))
    fout.close()
    frej.close()

def dump_users(users, outfile):
    """Write all the users to a file with their scores."""
    fout = open_maybe_gzip(outfile, 'w')
    for k, v in users.iteritems():
        mean = sum(v) / len(v)
        fout.write("%4f %s\n" % (mean, k))
    fout.close()


def queue_from_file(fn):
    from subprocess import check_call
    debug("Queueing users in %r for download" % (fn,))
    os.chdir(BASE_DIR) #because twextract has git-root relative imports
    f = open(fn)
    for line in f:
        score, user = line.strip().split(None, 1)
        check_call(['python', 'twextract/request_queue.py', user, '-c' 'config/ceres'])
        time.sleep(0.001)
    f.close()

def main():
    parser = OptionParser()
    parser.add_option("-m", "--trigram-mode", help="how to trigrammise [%s]" % DEFAULT_MODE,
                      default=DEFAULT_MODE)
    parser.add_option("-c", "--recompile", help="Derive trigrams from corpora", action="store_true")
    parser.add_option("-C", "--recompile-all", help="Derive trigrams for all modes and exit",
                      action="store_true")

    parser.add_option("-T", "--trial", help="show scores of tweets, not users", action="store_true")

    parser.add_option("-t", "--threshold", help="use this as threshold",
                      default=str(DEFAULT_THRESHOLD), metavar="(STRING|FLOAT)")

    parser.add_option("-i", "--input", help="input file or directory", metavar="PATH")
    parser.add_option("-b", "--bad-file", help="write rejects here", metavar="FILE")
    parser.add_option("-g", "--good-file", help="write good ones here", metavar="FILE")
    parser.add_option("-d", "--dump-file", help="write them all here, perhaps in order", metavar="FILE")
    parser.add_option("-q", "--queue", help="queue the good users for download", action="store_true")
    parser.add_option("-Q", "--queue-from-file",
                      help="queue from a pre-existing list (no evaluation)", metavar="FILE")
    parser.add_option("-r", "--report", help="get statistical data on stderr", action="store_true")

    parser.add_option("-f", "--offset-factor", help="English unseen trigram probablility factor",
                      type="float", default=TRIGRAM_OFFSET_FACTOR, metavar="FLOAT")
    parser.add_option("-a", "--anti-offset-factor", help="non-English unseen trigram probablility factor",
                      type="float", default=ANTI_TRIGRAM_OFFSET_FACTOR, metavar="FLOAT")


    (options, args) = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()

    if options.recompile_all:
        pre_cook(modes=MODES, corpi=CORPI + ANTI_CORPI)
        pre_cook_full(modes=MODES, corpi=CORPI + ANTI_CORPI)
        sys.exit()

    if options.queue_from_file:
        queue_from_file(options.queue_from_file)
        sys.exit()

    src = options.input
    good = options.good_file
    bad = options.bad_file
    dump = options.dump_file

    mode = options.trigram_mode

    if options.recompile:
        pre_cook_full(modes=[mode], corpi=CORPI + ANTI_CORPI)

    tg = get_trigram_with_antimodel(mode, tof=options.offset_factor,
                                    atof=options.anti_offset_factor)

    try:
        threshold = float(options.threshold)
    except ValueError:
        threshold = tg.probable_similarity(options.threshold)
        debug("Threshold from %r is %s" %(options.threshold, threshold))

    if options.trial:
        if src is None:
            src = TEST_FILE_1
        if good or bad:
            bisect_the_hose(tg, src, good, bad, threshold=threshold)
        if dump:
            order_the_hose(tg, src, dump)

    elif good or bad or dump:
        if src is None:
            src = STASH_DIR
        users = _group_by_user(tg, src)
        if options.report:
            users_report(users)
        if good or bad:
            partition_users(users, good, bad, threshold)
        if dump is not None:
            dump_users(users, dump)
        if good and options.queue:
            queue_from_file(good)
    else:
        debug("nothing much to do!")



if __name__ == '__main__':
    main()
