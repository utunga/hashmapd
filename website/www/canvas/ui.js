/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/** interpret_query puts query parameters in an object, slightly type-aware-ly
 *
 * Simply-typed attributes in the destination object (string, bool,
 * number) are replaced the equivalently named and typed query
 * parameters.  If query is undefined, the window.location url is
 * used.  If it is a string, it is parsed as a query string.  If it is
 * an object, it is used directly as a mapping.
 *
 * This feels completely dodgy to a server side programmer, but is OK
 * on the client side.  They can only mangle their own browsers.  Probably.
 *
 * @param dest is an object in which to put the query parameters
 * @param query is an optional query string or object.
 */

function interpret_query(dest, query){
    if (query === undefined || typeof(query) == 'string'){
        query = parse_query(query);
    }
    for (var param in query){
        if (param in dest){
            var v = query[param];
            var existing = dest[param];
            switch(typeof(existing)){
            case "number":
                v = parseFloat(v);
                if (! isNaN(v)){
                    dest[param] = v;
                }
                break;
            case "boolean":
                v = v.toLowerCase();
                dest[param] = (!(v == "0" ||
                                v == "false" ||
                                v == "no" ||
                                v == ""));
                break;
            case "string":
                dest[param] = v;
                break;
            default:
                log("ignoring " + param + "=" + v +
                    " (unknown type)");
            }
        }
        else {
            log("ignoring " + param + "=" + v +
                " (unknown attribute)");
        }
    }
}

/** parse_query turns an http query into a mapping object
 *
 * You would think this would be in a library, but no.
 *
 * @param query is a query string or undefined to use current url.
 * @return a mapping.
 */

function parse_query(query){
    if (query === undefined)
        query = window.location.search.substring(1);
    if (! query) return {};
    var args = {};
    var re = /([^&=]+)=?([^&]*)/g;
    while (true){
        var match = re.exec(query);
        if (match === null){
            return args;
        }
        args[decodeURIComponent(match[1])] = decodeURIComponent(match[2].replace(/\+/g, " "));
    }
}

/**construct_form makes a quick for for testing purposes
 */

function construct_form(){
    $("#helpers").append('<form id="state"></form>');
    var form = $("#state");
    for (var param in $state){
        var existing = $state[param];
        if (typeof(existing) in {number: 1, string: 1}){
            form.append(param + '<input name="' + param + '" value="' +
                        existing + '"><br>');
        }
    }

    var submit = function() {
        var q = form.serialize();
        set_state(q);
        return false;
    };

    form.append('<button>go</button>');
    form.submit(submit);
}

/** set_state redraws to match a query string or $state-like object
 *
 * If nothing changes, no redraw.
 *
 * @param data is an http query string or a mapping object.
 */

function set_state(data){
    if (typeof(data) == 'string'){
        data = parse_query(data);
    }
    
    var copy = {};
    for (k in $state){
        copy[k] = $state[k];
    }
    interpret_query($state, data);
    var h = window.history;
    var loc = window.location;
    var url = loc.href.split("?", 1)[0] + "?" +  $.param($state);

    if (url != loc.href){
        h.replaceState($state, "Hashmapd", url);
    }
    /*make sure something changed */
    for (k in $state){
        if (copy[k] != $state[k]){
            set_ui($state);
            hm_draw_map();
            return;
        }
    }
    log("$state unchanged by set_state");
}

/** set_sets the UI elements to match the given state
 *
 * @param state takes the form of the global $state.
 */

function set_ui(state){
    var slider = $("#zoom-slider");
    slider.slider("value", state.zoom);
    /*debug form */
    log("set_ui", slider.slider("value"));
    $("#state input").each(
        function(i) {
            //dump_object(this);
            var k = this.name;
            log(this, k, i);
            if (state[k] !== undefined){
                this.value = state[k];
            }
        }
    );
}

function construct_ui(){
    var slider = $("#zoom-slider");
    $(slider).slider({ orientation: 'vertical',
                       max: 6,
                       min: 0,
                       slide: function( event, ui ) {
                           set_state({'zoom': ui.value});
                       }
                     });
    slider.offset($($page.canvas).offset());
    var canvas = $($page.canvas);
    var x, y;
    canvas.mousedown(function(e){
        x = e.pageX;
        y = e.pageY;
    });
    var finish = function(e){
        if (x !== undefined && y !== undefined){
            pan_pixel_delta(e.pageX - x, e.pageY - y);
        }
        x = undefined;
        y = undefined;
    };
    canvas.mouseup(finish);
    canvas.mouseout(finish);
}

function pan_pixel_delta(dx, dy){
    var scale_x = $page.x_scale * (1 << $state.zoom);
    var scale_y = $page.y_scale * (1 << $state.zoom);
    var px = dx / scale_x;
    var py = dy / scale_y;
    var x = parseInt($state.x - px);
    var y = parseInt($state.y - py);
    x = Math.max($page.min_x, Math.min($page.max_x, x));
    y = Math.max($page.min_y, Math.min($page.max_y, y));
    set_state({x: x, y: y});
}
