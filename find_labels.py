#!/usr/bin/python

import sys, os
import heapq
#import json

def log(*args):
    for x in args:
        print >> sys.stderr, x,
    print >> sys.stderr

def get_row_emitter(filename):
    # let's NOT parse it properly, which would be slow and memory hungry.
    # Rows look like:
    #
    # {"key":["\u0640\ufbb1",1,3,0,2,3,3],"value":3},
    # {"key":["_",0,1,0,3,2,3],"value":1},
    f = open(filename)
    prefix = '{"key":["'
    suffix = '},'
    start = len(prefix)
    end = - len(suffix)
    for line in f:
        if not line.startswith(prefix):
            log("ignoring", repr(line))
            continue
        line = line.strip()[start: end]
        #log(start, end, line)
        key, value = line.rsplit('],"value":', 1)
        token, coords = key.rsplit('",', 1)
        value = int(value)
        #log (token, coords, value)
        yield (token, coords, value)
    f.close()


def save_as_text(locations, out_file, limit=100):
    keys = sorted(locations.keys())
    f = open(out_file, 'w')
    for k in keys:
        v = locations[k]
        print >> f, k + ':'
        v.sort()
        v.reverse()
        for rvalue, value, token in v[:limit]:
            print >> f, "%r, %0.4f, %d; " % (token, rvalue, value),
        print >> f

def save_as_json(locations, out_file, limit=100):
    keys = sorted(locations.keys())
    f = open(out_file, 'w')
    from math import log
    f.write('{"rows":[\n')
    for k in keys:
        v = locations[k]
        v.sort()
        v.reverse()
        for rvalue, value, token in v[:limit]:
            #adj = int(100 * rvalue * (5 + log(value)))
            adj = rvalue * 1000#int(100 * rvalue * (5 + log(value)))
            token = repr(token).replace(r'\\u', r'\u')
            s = '{"key":[%s,%s],"value":%d},\n' % (token, k, adj)
            f.write(s)
    f.write(']}\n')

def save_as_html(locations, out_file, limit=100):
    f = open(out_file, 'w')
    cells = [[[] for a in range(64)] for b in range(64)]
    for k, v in locations.iteritems():
        v.sort()
        v.reverse()
        for rvalue, value, token in v[:limit]:
            x = 0
            y = 0
            adj = int(rvalue)
            token = repr(token).replace(r'\\u', r'\u')
            for n in k.split(','):
                n = int(n)
                x = (x << 1) + (n & 1)
                y = (y << 1) + (n >> 1)
            cells[y][x].append((adj, token, k))

    f.write('<html><style>'
            'td.sea {background: #cef}'
            'td {font: 10px sans-serif; border: 1px solid #ccc; padding: 2px;}'
            '</style>\n'
            '<table style="font: 10px sans-serif">\n')
    for row in cells:
        f.write('<tr>\n')
        for cell in row:
            if cell:
                coords = cell[0][2]
                f.write('<td id="%s">\n' % coords.replace(',', ''))
                f.write("<b>%s</b><br/>" % (coords))

                #cell.sort()
                for tk in cell:
                    f.write("%s: %s<br/>" % (tk[1], tk[0]))
            else:
                f.write('<td class="sea">\n')

    f.write('</table></html>\n')


def main(json_file, out_file):
    from math import log

    rows = get_row_emitter(json_file)

    #need to sort by tokens first to get totals for each token
    tokens = {}
    for token, coords, value in rows:
        tokens.setdefault(token, []).append((value, coords))

    locations = {}
    for token, v in tokens.iteritems():
        total = float(sum(value for value, coords in v))
        logtotal = log(total) + 3
        for value, coords in v:
            locations.setdefault(coords, []).append((#value,
                                                     value / total * 1000,
                                                     #value / total * 100 * logtotal,
                                                     value, token))

    save_as_html(locations, out_file, 10)

main(sys.argv[1], sys.argv[2])

