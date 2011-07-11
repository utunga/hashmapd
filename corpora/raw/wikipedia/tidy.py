#!/usr/bin/python

import re, os, sys

def clean1(fn):
    f = open(fn)
    fw = open(fn + '.out', 'w')

    copying = 0
    for line in f:
        if '***** Contents *****' in line or 'Skip_to_table_of_contents' in line:
            copying = 1
        elif ('See also *****' in line or
              'Retrieved from "http://en.wikipedia.org/' in line or
              '***** Notes *****' in line):
            copying = 0
        elif line.startswith('*****') and copying == 1:
            copying = 2
        elif line.startswith('Jump to: navigation') and '-talk' not in fn:     
            copying = 2
        elif copying == 2 and not line[0] in '[*':
            fw.write(line)
            

for fn in os.listdir('.'):
    if fn.endswith('.txt'):
        clean1(fn)
