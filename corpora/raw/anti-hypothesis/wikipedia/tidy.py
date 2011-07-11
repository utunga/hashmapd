#!/usr/bin/python

import re, os, sys



def clean1(fn):
    f = open(fn)
    fw = open(fn + '.out', 'w')

    ref_re = re.compile(r'^\d+\. ')

    copying = 2
    wait = 0
    for line in f:
        line = line.strip()
        if line.startswith('[search              ]'):
            break
        if ref_re.match(line):
            wait += 3
        if wait == 0:
            fw.write(line + '\n')
        else:
            wait -= 1
            

for fn in os.listdir('.'):
    if fn.endswith('.txt'):
        clean1(fn)
