#!/usr/bin/python

import sys, os
import heapq
import gzip
import json
import cPickle
from operator import itemgetter
from collections import namedtuple


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
            continue
        line = line.strip()[start:]
        key, value = line.rsplit('],"value":', 1)
        if has_token:
            token, coords = key.rsplit('",', 1)
            token = token.decode('unicode_escape').encode('utf-8')
        else:
            token, coords = None, key
        value = int(value.rsplit('}', 1)[0])
        yield (token, coords, value)
    f.close()

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

def empty_cell(location=None):
    return {'data':[],
            'tokens': frozenset(),
            'classes': [],
            'users': '',
            'location': location,
            }

def make_location_cells(locations, orderby, users=None, limit=10):
    log("making location cells")
    coord = locations.iterkeys().next()
    cells = {}
    index = {}
    from math import log as mlog
    for k, v in locations.iteritems():
        if not k in index:
            cell = empty_cell(k)
            index[k] = cell
            cells[quad_to_xy(k)] = cell
        else:
            cell = index[k]
        v.sort()
        v.reverse()
        for value, overall, token in v[:limit]:
            if orderby == 'adjusted':
                adj = int((10000.0 * value) / overall * (4 + mlog(overall)))
            elif orderby == 'total':
                adj = value
            else:
                adj = int(10000.0 * value / overall)
            cell['data'].append((adj, token))

    for cell in index.values():
        cell['data'].sort()
        cell['data'].reverse()
        cell['tokens'] = frozenset(k for v, k in cell['data'])


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


def save_as_json(index, out_file, limit=2):
    f = open(out_file, 'w')
    rows = []
    for coords, cell in index.iteritems():
        coords = [int(x) for x in coords.split(',')]
        for value, token in cell['data'][:limit]:
            key = [token.decode('utf-8')] + coords
            rows.append({'key': key, 'value': value})

    json.dump({"rows": rows}, f, separators=(',', ':'), indent=None)
    f.close()

def save_as_html(cells, out_file):
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

    for cell in cells.itervalues(): # fake for loop to access arbitrary item
        coords = cell['location']
        precision = len(coords.split(','))
        size = 1 << precision
        break

    for y in xrange(size):
        f.write('<tr>\n')
        for x in xrange(size):
            cell = cells.get((x, y))
            if cell is None:
                f.write('<td class="sea">\n')
                continue
            if cell['data']:
                coords = cell['location']
                f.write('<td id="%s" class="%s">' % (coords, ' '.join(cell.get('classes', ()))))
                f.write('<span class="users">%s</span>' % (cell.get('users', '')))
                f.write('<a href="#%s">%s</a><br/>' % (coords, coords))
                for tk in cell['data']:
                    f.write("%s: %s<br/>" % (tk[1], tk[0]))
            elif cell.get('users'):
                f.write('<td class="shallows"><span class="users">%s</span>\n' % (cell.get('users', '')))
            else:
                log("you should never see this unreachable message about the misformed cell ",
                    (x, y), cell)
                f.write('<td class="error"> %s' % cell)


    f.write('</table></html>\n')


def add_users(index, cells, users):
    if users:
        for token, coords, value in users:
            cell = index.get(coords, empty_cell())
            if cell['location'] is None:
                cell['location'] = coords
                cells[quad_to_xy(coords)] = cell
            cell['users'] = value


def get_cached_data():
    log("unpickling rows, data")
    f = open(PICKLE)
    tokens, users = cPickle.load(f)
    f.close()
    return tokens, users

def save_cached_data(tokens, users):
    log("generating cache")
    f = open(PICKLE, 'w')
    cPickle.dump((tokens, list(users)), f)
    f.close()

def really_get_data(token_json, user_json):
    log("parsing json")
    users = get_row_emitter(user_json, False)
    rows = get_row_emitter(token_json)
    tokens = group_by_token(rows)
    return tokens, list(users)

def get_data(token_json, user_json, use_cache=USE_CACHE):
    if use_cache:
        try:
            tokens, users = get_cached_data()
        except IOError:
            tokens, users = really_get_data(token_json, user_json)
            save_cached_data(tokens, users)
    else:
        tokens, users = really_get_data(token_json, user_json)
    return tokens, users


def by_location(token_json, user_json, out_file, orderby, use_cache=USE_CACHE):
    tokens, users = get_data(token_json, user_json, use_cache=use_cache)
    locations = calc_locations(tokens)
    del tokens
    log("making cells")
    cells, index = make_location_cells(locations, orderby, 10)
    add_users(index, cells, users)

    log("finding common tokens")
    find_common_tokens(cells)
    save_as_html(cells, out_file)


def make_token_cells(tokens, orderby, limit=100):
    log("making token cells")
    cells = {}
    index = {}
    #{token: [(value, coord), (value, coord),...]}
    for token, locations in tokens.iteritems():
        if orderby == 'token_relative':
            mul = 1000.0 / sum(a for a, b in locations)
        else:
            mul = 1
        locations.sort()
        max_value = locations[-1][0]
        for v, coord in reversed(locations):
            if v != max_value:
                break
            if coord in index:
                cell = index[coord]
            else:
                cell = empty_cell(coord)
                index[coord] = cell
                cells[quad_to_xy(coord)] = cell
            cell['data'].append((int(v * mul), token))

    for cell in index.itervalues():
        cell['data'].sort()
        cell['data'].reverse()
        cell['data'] = cell['data'][:limit]
        cell['tokens'] = frozenset(t for v, t in cell['data'])
    return cells, index


def by_token(token_json, user_json, out_file, orderby, use_cache=USE_CACHE):
    tokens, users = get_data(token_json, user_json, use_cache=use_cache)
    cells, index = make_token_cells(tokens, orderby)
    add_users(index, cells, users)
    find_common_tokens(cells)
    save_as_html(cells, out_file)

def by_magic_heuristic(token_json, user_json, out_file, orderby, use_cache=USE_CACHE, limit=10):
    tokens, users = get_data(token_json, user_json, use_cache=use_cache)
    locations = calc_locations(tokens)
    cells = {}
    index = {}

    cells, index = make_location_cells(locations, 'total', limit)
    t_cells, t_index = make_token_cells(tokens, 'token_total', limit)

    #cells, index should be complete; t_* not so,
    # so import the latter into the former
    for k, cell in index.iteritems():
        if k in t_index:
            tcell = t_index[k]
            # data for each goes [(value, token)]
            # to combine them, we sort with a schwartzian transform
            # so identical tokens will be adjacent
            everything = [(a, b) for b, a in cell['data'] + tcell['data']]
            everything.sort()
            combo = []
            token = everything[0][0]
            value = 0
            for k, v in everything:
                if k == token:
                    #accumulate before adjusting
                    value += v
                    continue
                combo.append((value, token))
                token = k
                value = v
            cell['data'] = combo

    for k, cell in index.iteritems():
        d = cell['data']
        fixed = []
        for value, token in d:
            #some heuristic adjustments
            if len(token) > 8:
                value = value * 8 / len(token)
            if token.startswith('@'):
                value = value / 10
            elif token.startswith('http://'):
                value = value / 10
            if value > 0:
                fixed.append((value, token))

        fixed.sort()
        fixed.reverse()
        cell['data'] = fixed[:limit * 2]
        cell['tokens'] = frozenset(t for v, t in cell['data'])

    add_users(index, cells, users)
    find_common_tokens(cells)
    save_as_json(index, out_file[:-5] + '.json')
    save_as_html(cells, out_file)



USER_JSON = 'user-count.json'

try:
    orderby = sys.argv[4]
    token_json = sys.argv[1]
    user_json = sys.argv[2]
    out_file = sys.argv[3]
    if orderby in ('magic', ):
        by_magic_heuristic(token_json, user_json, out_file, orderby)
    elif orderby in ('token_total', 'token_relative'):
        by_token(token_json, user_json, out_file, orderby)
    else:
        by_location(token_json, user_json, out_file, orderby)
except IndexError:
    print "USAGE %s tokens.json outputfile {total, relative, adjusted}" % sys.argv[0]


