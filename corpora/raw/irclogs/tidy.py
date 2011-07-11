#!/usr/bin/python

import re, os, sys
import gzip

def clean1(fn):
    if fn.endswith('.gz'):
        f = gzip.open(fn)
        fn = fn[:-3]
    else:
        f = open(fn)
    fw = open(fn + '.out', 'w')
    copying = 0
    for line in f:        
        #15:33 < BioTube>
        #15:33  * PsyForce checks version...
        try:
            nick, msg = line.split('> ')
        except ValueError:
            continue
        if not msg.startswith('*.split'):
            fw.write(msg)
            

for fn in os.listdir('.'):
    if fn.endswith('.log') or fn.endswith('.log.gz'):
        clean1(fn)
