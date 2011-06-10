/* Sum the counts for identical terms */
function (keys, values, rereduce){
    var out = [];
    var i;
    if (rereduce){
        var v = values[0];
        for (i = 1; i < values.length; i++){
            v.concat(values[i]);
        }
        values = v;
    }
    values.sort();
    var term = values[0][0];
    var count = values[0][1];
    for (i = 1; i < values.length; i++){
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
