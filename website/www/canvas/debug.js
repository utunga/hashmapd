/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 *
 * This file has debug and timer code.
 */

/** get_timer returns a namespace for timing code
 *
 * Note that it is essentially a singleton -- to avoid the tedium of
 * javascript's scoping of "this" in async contexts.
 */
function get_timer(){
    var times = [];
    return {
        times: times,
        checkpoint: function(label){
            times.push([Date.now(), label]);
        },
        time_func: function(func){
            /*arguments is not real array, no .slice or .shift, so you
             *need to slice by copying.*/
            var args = [];
            for (var i = 1; i < arguments.length; i++){
                args.push(arguments[i]);
            }
            checkpoint("start " + arguments.callee.name);
            var r = func.apply(args);
            checkpoint("finish " + arguments.callee.name);
            return r;
        },
        results: function(){
            var s = '<table><tr><td colspan="3">milliseconds<tr><td><td>time<td>delta';
            var t2 = 0;
            for (var i = 0; i < times.length; i++){
                var t = times[i][0] - times[0][0];
                var d = t - t2;
                t2 = t;
                s += "<tr><td>" + times[i][1] + "<td>" + t + "<td><b>" + d + "\n";
            }
            s += "</table>";
            $("#debug").append(s);
        }
    };
}

function log(){
    var s = "<div>";
    for (var i = 0; i < arguments.length; i++){
        s += arguments[i] + " ";
    }
    s += "</div>";
    $("#debug").append(s);
};