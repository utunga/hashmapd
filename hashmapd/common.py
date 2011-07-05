from subprocess import Popen, PIPE
import gzip

def find_git_root():
    p = Popen(["git", "rev-parse", "--show-toplevel"], stdout=PIPE)
    root = p.communicate()[0].strip()
    if p.returncode:
        raise SystemExit("This does not seem to be a git repository (git returned %s)" %
                         p.returncode)
    return root


def open_maybe_gzip(filename, mode='rb'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)
