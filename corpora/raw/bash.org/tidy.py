#!/usr/bin/python

import re, os, sys

def clean1(fn):
    f = open(fn)
    fw = open(fn + '.out', 'w')

    copying = 0
    for line in f:
        # nick separator varies on bash.org
        for sep in '>)]':
            bits = line.decode('latin-1').encode('utf8').split(sep, 1)
            if len(bits) == 2:
                msg = bits[1].lstrip()
                break
        else:
            continue
        if msg and msg[0] not in '+([/>-|=0':
            fw.write(msg)
            

for fn in os.listdir('.'):
    if fn.endswith('.txt'):
        clean1(fn)
