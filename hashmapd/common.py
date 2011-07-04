from subprocess import Popen, PIPE


def find_git_root():
    p = Popen(["git", "rev-parse", "--show-toplevel"], stdout=PIPE)
    root = p.communicate()[0].strip()
    if p.returncode:
        raise SystemExit("This does not seem to be a git repository (git returned %s)" %
                         p.returncode)
    return root
