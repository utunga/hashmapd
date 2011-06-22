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

def make_cells(locations, limit=100):
    cells = [[{'data':[],
               'tokens': set(),
               'classes': []
              } for a in range(64)] for b in range(64)]
    log("making cells, sets")
    for k, v in locations.iteritems():
        v.sort()
        v.reverse()
        for rvalue, value, token in v[:limit]:
            x = 0
            y = 0
            adj = int(rvalue)
            token = repr(token).replace(r'\\u', r'\u').decode('unicode_escape').encode('utf-8')
            for n in k.split(','):
                n = int(n)
                x = (x << 1) + (n & 1)
                y = (y << 1) + (n >> 1)
            cell = cells[y][x]
            cell['location'] = k
            cell['data'].append((adj, token))
            cell['tokens'].add(token)
    return cells

def find_common_tokens(cells):
    log("associating sets")
    for y in range(1, len(cells)):
        for x in range(1, len(cells[0])):
            cell = cells[y][x]
            up = cells[y - 1][x]
            left = cells[y][x - 1]
            if cell['tokens'] & left['tokens']:
                cell['classes'].append('left')
                left['classes'].append('right')
            if cell['tokens'] & up['tokens']:
                cell['classes'].append('up')
                up['classes'].append('down')
    return cells

def save_as_html(locations, out_file, limit=100):
    f = open(out_file, 'w')
    cells = make_cells(locations, limit)
    find_common_tokens(cells)

    log("writing html")
    f.write('<html><meta http-equiv="Content-Type" content="text/html;charset=UTF-8">'
            '<style>'
            'td.sea {background: #cef}'
            'td {font: 10px sans-serif; border: 1px solid #ccc; padding: 2px;}'
            'a {font-weight: bold; text-decoration: none;}'
            '.up, .down, .left, .right {background: #fd9}'
            '.up {border-top: 1px #fe0 solid}'
            '.down {border-bottom: 1px #fe0 solid}'
            '.left {border-left: 1px #fe0 solid}'
            '.right {border-right: 1px #fe0 solid}'
            '</style>\n'
            '<table style="font: 10px sans-serif">\n')
    for row in cells:
        f.write('<tr>\n')
        for cell in row:
            if cell['data']:
                coords = cell['location']
                ID = coords#.replace(',', '')
                f.write('<td id="%s" class="%s">\n' % (ID, ' '.join(cell['classes'])))
                f.write('<a href="#%s">%s</a><br/>' % (ID, coords))

                #cell.sort()
                for tk in cell['data']:
                    f.write("%s: %s<br/>" % (tk[1], tk[0]))
            else:
                f.write('<td class="sea">\n')

    f.write('</table></html>\n')


def main(json_file, out_file, style):
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
            locations.setdefault(coords, []).append(({'total': value,
                                                      'relative': value / total * 1000,
                                                      'adjusted': value / total * 100 * logtotal,
                                                      }[style],
                                                     value, token))

    save_as_html(locations, out_file, 10)

main(sys.argv[1], sys.argv[2], sys.argv[3])

