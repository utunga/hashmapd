/*XXX This does not work at high levels of grouping!
 *
 * For example, with group_level=3, you get:
 *
 * [3, 1, 0] -> [[["They", 125492], null], [["They", 75388], null]]
 *
 * (that should be [["They", 200880]], with no nulls in sight).
 *
 * I can't see why.
 */

function (keys, values, rereduce){
    var out = [];
    values.sort();
    var term = values[0][0];
    var count = values[0][1];
    for (var i = 1; i < values.length; i++){
        var next = values[i];
        if (next[0] == term){
            count += next[1];
        }
        else {
            out.push([term, count]);
            term = next[0];
            count = next[1];
        }
    }
    if (out.length == 0 || term != out[out.length - 1][0]){
        out.push([term, count]);
    }
    return out;
};
