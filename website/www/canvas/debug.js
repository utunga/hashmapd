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
    if (! $const.DEBUG){
        return function(){};
    }
    var previous = Date.now();
    var start = previous;
    var s = ('<table id="hm_timer"><tr><td colspan="3">milliseconds<tr>' +
             '<td><td>time<td>delta</table>');
    $("#debug").append(s);
    var table = $("#hm_timer");
    return function(label, reset){
        var now = Date.now();
        if (reset){
            start = now;
        table.append('<tr><td colspan="2">....................</tr>');
        }
        var t = now - start;
        var d = now - previous;
        table.append("<tr><td>" + label + "<td>" + t +
                     "<td><b>" + d + "</b></tr>");
        previous = now;
    };
}

function log(){
    if (! $const.DEBUG){
        return;
    }
    var s = "<div>";
    for (var i = 0; i < arguments.length; i++){
        s += arguments[i] + " ";
    }
    s += "</div>";
    $("#debug").append(s);
};

function dump_object(o, depth, padding){
    if(padding === undefined)
        padding = '';
    if (depth === undefined)
        depth = 1;
    var s = padding + '{\n';
    for (var x in o){
        if (typeof(o[x]) == 'object' && depth > 1){
            s += dump_object(o[x], depth - 1, padding + '  ');
        }
        s += padding + '  ' + x + ':' + o[x] + ',\n';
    }
    s += padding + '}';
    if (padding == ''){
        log(s);
    }
    return s;
}

function loading_screen(){
    var outer = $("#loading");
    outer.css("display", "block");
    outer.css("visibility", "visible");
    outer.css("z-index", "2");

    var div = outer.append("<div>wheee</div>");

    return {
        show: function(text){
            div.html(text);
            log("set loader to " + div.html());
        },
        done: function(text){
            outer.css("display", "none");
            outer.css("visibility", "hidden");
        },
        tick: function(){
            div.html(div.html() + ".");
            log("set loader to " + div.html());
        }

    };
}
