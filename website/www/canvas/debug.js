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
    var previous = Date.now();
    var start = previous;
    var s = ('<table id="hm_timer"><tr><td colspan="3">milliseconds<tr>' +
             '<td><td>time<td>delta</table>');
    $("#debug").append(s);
    var table = $("#hm_timer");
    var checkpoint = function(label){
        var now = Date.now();
        var t = now - start;
        var d = now - previous;
        table.append("<tr><td>" + label + "<td>" + t +
                     "<td><b>" + d + "</b></tr>");
        previous = now;
    };
    return {
        checkpoint: checkpoint,
        time_func: function(func){
            /*arguments is not real array, no .slice or .shift, so you
             *need to slice by copying.*/
            var args = [];
            for (var i = 1; i < arguments.length; i++){
                args.push(arguments[i]);
            }
            checkpoint("start " + func.name);
            var r = func.apply(undefined, args);
            checkpoint("finish " + func.name);
            return r;
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