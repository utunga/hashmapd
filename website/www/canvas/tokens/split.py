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

    f = open(filename)
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


def start_f(ofn):
    of = open(ofn, "w")
    of.write('{"rows":[\n')
    return of

def end_f(of, row):
    of.write('{"key":["%s",%s],"value":%s}\n' % row)
    of.write(']}\n')
    of.close()

def split_json(fn, ofn, size=100000):
    endtoken = None
    remaining = size
    of = start_f(ofn % 1)
    fileno = 1
    rows = get_row_emitter(fn)
    prevrow = rows.next()
    for row in rows:
        token = row[0]
        remaining -= 1
        if remaining == 0:
            endtoken = token
            of.write('{"key":["%s",%s],"value":%s},\n' % prevrow)

        elif remaining < 0 and token != endtoken:
            end_f(of, prevrow)
            fileno += 1
            of = start_f(ofn % fileno)
            remaining = size

        else:
            of.write('{"key":["%s",%s],"value":%s},\n' % prevrow)

        prevrow = row




    end_f(of, row)

split_json('all-tokens-8.json', 'all-tokens-8-part-%03d.json')
