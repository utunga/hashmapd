#!/usr/bin/python

import sys, os
import heapq
import gzip
#import json
import cPickle
from operator import itemgetter

USER_JSON = 'user-count.json'
PICKLE = 'labels.pickle'
USE_CACHE=True

def log(*args):
    for x in args:
        print >> sys.stderr, x,
    print >> sys.stderr

def open_maybe_gzip(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename)
    else:
        return open(filename)

def get_row_emitter(filename, has_token=True):
    # let's NOT parse it properly, which would be slow and memory hungry.
    # (honestly, the 6 coordinate file is a 160MB file, and this already uses > 1 GB).
    # Rows look like:
    #
    # {"key":["\u0640\ufbb1",1,3,0,2,3,3],"value":3},
    # {"key":["_",0,1,0,3,2,3],"value":1},
    #
    # and for user count:
    # {"key":[0,1,0,3,1,1],"value":1},

    f = open_maybe_gzip(filename)
    if has_token:
        prefix = '{"key":["'
    else:
        prefix = '{"key":['
    suffix = '},'
    start = len(prefix)
    end = - len(suffix)
    for line in f:
        if not line.startswith(prefix):
            #log("ignoring", repr(line))
            continue
        line = line.strip()[start:]
        #log(start, end, line)
        key, value = line.rsplit('],"value":', 1)
        if has_token:
            token, coords = key.rsplit('",', 1)
        else:
            token, coords = None, key
        token = repr(token).replace(r'\\u', r'\u').decode('unicode_escape').encode('utf-8')
        #log(line)
        value = int(value.rsplit('}', 1)[0])
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
            adj = rvalue * 1000#int(100 * rvalue * (5 + log(value)))
            token = repr(token).replace(r'\\u', r'\u')
            s = '{"key":[%s,%s],"value":%d},\n' % (token, k, adj)
            f.write(s)
    f.write(']}\n')

def group_by_token(rows):
    log("group_by_token")
    tokens = {}
    for token, coords, value in rows:
        tokens.setdefault(token, []).append((value, coords))
    return tokens

def calc_locations(tokens):
    log("calculating locations")
    locations = {}
    for token, v in tokens.iteritems():
        total = float(sum(value for value, coords in v))
        for value, coords in v:
            locations.setdefault(coords, []).append((value, total, token))
    return locations

def quad_to_xy(coords):
    x = 0
    y = 0
    for n in coords.split(','):
        n = int(n)
        x = (x << 1) + (n & 1)
        y = (y << 1) + (n >> 1)
    return (x, y)

def make_location_cells(locations, orderby, users=None, limit=10):
    log("making cells, sets")
    coord = locations.iterkeys().next()
    cells = {}
    index = {}
    from math import log as mlog
    for k, v in locations.iteritems():
        if not k in index:
            cell = {'data':[],
                    'tokens': set(),
                    'classes': [],
                    'users': '',
                    'location': k,
                    }
            index[k] = cell
            cells[quad_to_xy(k)] = cell
        else:
            cell = index[k]
        v.sort()
        v.reverse()
        for value, overall, token in v[:limit]:
            if orderby == 'adjusted':
                adj = int(1000.0 * value / overall * (5 + mlog(overall)))
            elif orderby == 'total':
                adj = value
            else: #elif orderby == 'relative':
                adj = int(1000.0 * value / overall)
            cell['data'].append((adj, token))
            cell['tokens'].add(token)

    for cell in index.values():
        cell['data'].sort()
        cell['data'].reverse()

    return cells, index

def find_common_tokens(cells):
    log("associating sets")
    for k, cell in cells.iteritems():
        x, y = k
        up = cells.get((x, y - 1))
        left = cells.get((x - 1,y))
        if left and cell['tokens'] & left['tokens']:
            cell['classes'].append('left')
            left['classes'].append('right')
        if up and cell['tokens'] & up['tokens']:
            cell['classes'].append('up')
            up['classes'].append('down')
    return cells

def save_as_html(cells, out_file, orderby):
    f = open(out_file, 'w')
    log("writing html")
    f.write('<html><meta http-equiv="Content-Type" content="text/html;charset=UTF-8">'
            '<style>'
            'td.sea {background: #cef; min-width: 0}'
            'td.error {background: #f00; min-width: 0}'
            'td.shallows {background: #eff; min-width: 0}'
            'td {font: 10px sans-serif; border: 1px solid #ccc; padding: 2px; min-width: 75px; vertical-align: top}'
            'a {font-weight: bold; text-decoration: none;}'
            '.up, .down, .left, .right {background: #feb}'
            '.up {border-top: 2px #fe0 solid}'
            '.down {border-bottom: 2px #fe0 solid}'
            '.left {border-left: 2px #fe0 solid}'
            '.right {border-right: 2px #fe0 solid}'
            '.users {color: #aaa; font-size: 20px; float: right}'
            '</style>\n'
            '<table style="font: 10px sans-serif">\n')
    for row in cells:
        f.write('<tr>\n')
        for cell in row:
            if cell['data']:
                coords = cell['location']
                ID = coords#.replace(',', '')
                f.write('<td id="%s" class="%s">' % (ID, ' '.join(cell['classes'])))
                f.write('<span class="users">%s</span>' % (cell['users']))
                f.write('<a href="#%s">%s</a><br/>' % (ID, coords))

                #cell.sort()
                for tk in cell['data']:
                    f.write("%s: %s<br/>" % (tk[1], tk[0]))
            elif cell['users']:
                f.write('<td class="shallows"><span class="users">%s</span>\n' % (cell['users']))
            else:
                f.write('<td class="sea">\n')

    f.write('</table></html>\n')


def get_cached_data():
    log("unpickling rows, data")
    f = open(PICKLE)
    rows, users = cPickle.load(f)
    locations = calc_locations(rows)
    f.close()
    return rows, users

def save_cached_data(rows, users):
    log("generating cache")
    f = open(PICKLE, 'w')
    cPickle.dump((list(rows), list(users)), f)
    f.close()
    return rows, users


def get_data(json_file, use_cache=True):
    if use_cache:
        try:
            rows, users = get_cached_data()
        except IOError:
            users = list(get_row_emitter(USER_JSON, False))
            rows = list(get_row_emitter(json_file))
            save_cached_data(rows, users)
    else:
        log("parsing json")
        users = get_row_emitter(USER_JSON, False)
        rows = get_row_emitter(json_file)
    return rows, users

def by_location(json_file, out_file, orderby, use_cache=False):
    rows, users = get_data(json_file, use_cache)
    locations = calc_locations(rows)
    log("making cells")
    cells = make_cells(locations, orderby, users, 10)

    log("finding common tokens")
    find_common_tokens(cells)

    save_as_html(cells, out_file, orderby)

def by_token(json_file, out_file, orderby, use_cache=False):
    rows, users = get_data(json_file, use_cache)
    cells, index = make_empty_cells(users)
    tokens = group_by_token(rows)
    #{token: [(value, coord), (value, coord),...]}
    used_cells = []
    for token, locations in tokens.iteritems():
        token = repr(token).replace(r'\\u', r'\u').decode('unicode_escape').encode('utf-8')
        if orderby == 'token_relative':
            mul = 1000.0 / sum(a for a, b in locations)
        else:
            mul = 1

        locations.sort()
        max_value = locations[-1][0]
        for v, coord in reversed(locations):
            if v != max_value:
                break
            cell = index[coord]
            used_cells.append(cell)
            cell['location'] = coord
            cell['tokens'].add(token)
            cell['data'].append((int(v * mul), token))

    for cell in used_cells:
        cell['data'].sort()
        cell['data'].reverse()

    find_common_tokens(cells)
    save_as_html(cells, out_file, orderby)


try:
    orderby = sys.argv[3]
    if orderby in ('token_total', 'token_relative'):
        by_token(sys.argv[1], sys.argv[2], orderby)
    else:
        by_location(sys.argv[1], sys.argv[2], orderby)
except IndexError:
    print "USAGE %s tokens.json outputfile {total, relative, adjusted}" % sys.argv[0]
