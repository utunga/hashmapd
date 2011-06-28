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

/**construct_msgbox makes a content editable box that recieves click
 * quadtree coordinates.
 */
function construct_msgbox(){
    $("#helpers").append('<h2>Click coordinates</h2><div id="click-coords"'
                         + 'contenteditable="true"></div>');
}

/**construct_form makes a quick for for testing purposes
 */
function construct_form(object, id, submit_func){
    $("#helpers").append('<h2>' + id + '</h2>' + '<form id="' + id +
                         '"><table></table></form>');
    var form = $("#" + id);
    var table = $("#" + id + " table");
    var param;
    for (param in object){
        var existing = object[param];
        switch(typeof(existing)){
        case "number":
        case "string":
            table.append('<tr><td>' + param + '<td><input name="' + param + '" value="' +
                         existing + '"></tr>');
            break;
        case "boolean":
            table.append('<tr><td>' + param + '<td>'
                         + '<input type="radio" value="true" title="true" name="'
                         + param + '"' + (existing ? " checked" : '') + '>'
                         + '<input type="radio" value="false" title="false" name="'
                         + param + '"' + (existing ? '' : " checked") + '>'
                         + '</tr>'
                        );

        }
    }
    table.append('<tr><td colspan="2"><button>go</button>');
    if (submit_func)
        form.submit(submit_func);
    return form;
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
    //dump_object(data);
    var copy = {};
    for (k in $state){
        copy[k] = $state[k];
    }
    if (data.zoom !== undefined){
        data.zoom = Math.min($const.MAX_ZOOM, Math.max(0, data.zoom));
    }

    interpret_query($state, data);
    var h = window.history;
    var loc = window.location;
    var url = loc.href.split("?", 1)[0] + "?" +  $.param($state);

    if (url != loc.href){
        /* a change in search term merits an addition in the page history,
         * but merely zooming or panning does not.
         *
         * XXX this actually interacts a little strangely with the first
         * loaded url, which does not get stored in history; and it ends up
         * storing the last view of each search term, not the first.
         */
        if (copy['token'] != $state['token']){
            h.pushState($state, "Hashmapd", url);
        }
        else{
            h.replaceState($state, "Hashmapd", url);
        }
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

/** set_ui sets the UI elements to match the given state
 *
 * @param state takes the form of the global $state.
 */

function set_ui(state){
    var slider = $("#zoom-slider");
    slider.slider("value", state.zoom);
    /*debug form */
    $("#state input").each(
        function(i) {
            var k = this.name;
            if (state[k] !== undefined){
                this.value = state[k];
            }
        }
    );
}

/** construct_ui makes a zoom slider and hooks up the token search box
 */

function construct_ui(){
    var slider = $("#zoom-slider");
    $(slider).slider({ orientation: 'vertical',
                       max: $const.MAX_ZOOM,
                       min: 0,
                       slide: function( event, ui ) {
                           set_state({'zoom': ui.value});
                       }
                     });

    var offset = $($page.canvas).offset();
    $("#zoom-controls").offset(offset);
    $("#zoom-out-button").click(function(){set_state({zoom: $state.zoom - 1})});
    $("#zoom-in-button").click(function(){set_state({zoom: $state.zoom + 1})});


    $("#token_form").submit(
        function(e){
            e.preventDefault();
            e.stopPropagation();
            var data = $("#token_input").val() || '';
            data = sanitise_token_input(data);
            $("#token_input").val(data);
            set_state({token: data});
            return false;
        }
    );
    $("#token_input").val($state.token);

}

/** normalise_token puts the token in the form expected by the backend.
 *
 * Most words are capitalised, but if you write with some caps, we'll
 * assume you mean that.
 */
function normalise_token(token){
    var lc = token.toLowerCase();
    if (lc == token){
        var uc = token.toUpperCase();
        token = uc.charAt(0) + lc.substr(1);
    }
    return token;
}

function sanitise_token_input(input){
    var tokens = input.trim().split(/\s+/, 4);
    var result;
    if (tokens.length == 3 && tokens[1] in $const.DENSITY_OPS){
        /*XXX should really have a proper parser */
        result = [normalise_token(tokens[0]),
                  tokens[1],
                  normalise_token(tokens[2])];
    }
    else if (tokens.length == 2){
        if (tokens[0] in $const.DENSITY_UNO_OPS){
            /* a unary operator on tokens[1] */
            result = [tokens[0], normalise_token(tokens[1])];
        }
        else if (tokens[0].match(/^>\d+$/)){
            /* a limit on tokens[1] */
            result = [tokens[0], normalise_token(tokens[1])];
        }
        else { /*two text tokens; give them an arbitrary operator */
            result = [normalise_token(tokens[0]),
                      '*',
                      normalise_token(tokens[1])];
        }
    }
    else {
        result = [normalise_token(tokens[0])];
    }
    result = result.join(' ');
    log("converted", input, "to", result);
    return result;
}


/** enable_drag sets up mouse dragging.
 *
 * It encloses a few event handlers which want to share state.
 */
function enable_drag(){
    var x, y;
    var drag_x, drag_y;
    var p = $page; /*local reference purely to make emacs js2-mode happy */

    /*ui_grabber is a div that floats above everything, grabbing mouse moves.
     * The point is to not worry about which canvas is on top.
     */
    var ui_grabber = $("#ui-grabber");
    var offset = $($page.canvas).offset();
    ui_grabber.offset(offset);
    ui_grabber.width($const.width);
    ui_grabber.height($const.height);

    var drag = function(e){
        p.mouse_dx = e.pageX - x;
        p.mouse_dy = e.pageY - y;
    };

    var start = function(e){
        x = e.pageX;
        y = e.pageY;
        ui_grabber.mousemove(drag);
        if ($const.DEBUG){
            var cx = e.pageX - offset.left;
            var cy = e.pageY - offset.top;
            var z = (1 << $state.zoom);
            var scale_x = $page.x_scale * z;
            var scale_y = $page.y_scale * z;
            var px = parseInt(cx / scale_x) + $page.min_x;
            var py = parseInt(cy / scale_y) + $page.min_y;
            var cc = $("#click-coords");
            cc.text(cc.text() + "\n[" + encode_point(px, py) + "]");
        }
        e.preventDefault();
        e.stopPropagation();
        return false;
    };

    var finish = function(e){
        if (x !== undefined && y !== undefined){
            pan_pixel_delta(e.pageX - x, e.pageY - y);
        }
        x = undefined;
        y = undefined;
        drag_x = undefined;
        drag_y = undefined;
        p.mouse_dx = 0;
        p.mouse_dy = 0;
        ui_grabber.mousemove(undefined);
    };

    var dblclick = function(e){
        e.preventDefault();
        e.stopPropagation();
        var dx =  e.pageX - offset.left - ($const.width / 2);
        var dy =  e.pageY - offset.top  - ($const.height / 2);
        log(dx, dy);
        pan_pixel_delta(-dx, -dy, 1);
        return false;
    };

    ui_grabber.mousedown(start);
    ui_grabber.mouseup(finish);
    ui_grabber.mouseout(finish);
    ui_grabber.dblclick(dblclick);

    if ($const.KEY_CAPTURE){
        $(document).keydown(key_events);
    }
}

function key_events(e){
    switch(e.keyCode){
    case 37:/*left*/
        pan_pixel_delta($const.width / 2, 0, 0);
        break;
    case 38:/*up*/
        pan_pixel_delta(0, $const.height / 2, 0);
        break;
    case 39:/*right*/
        pan_pixel_delta( -$const.width / 2, 0, 0);
        break;
    case 40:/*down*/
        pan_pixel_delta(0, -$const.height / 2, 0);
        break;
    case 107:/* + keypad */
    case 187:/* = */
        pan_pixel_delta(0, 0, 1);
        break;
    case 109:/* - keypad */
    case 189:/* - */
        pan_pixel_delta(0, 0, -1);
        break;
    }
}

function temp_pan_delta(dx, dy){
    log("mouse_move", dx, dy);

}

function hm_tick(){
    if ($page.mouse_dx && $page.mouse_dy){
        /* redraw the canvas on the temp canvas. */
        var z = (1 << $state.zoom);
        var dx = - $page.mouse_dx / ($page.x_scale * z);
        var dy = - $page.mouse_dy / ($page.y_scale * z);
        var d = get_zoom_pixel_bounds($state.zoom, $state.x + dx, $state.y + dy);
        var d2 = get_zoom_pixel_bounds($state.zoom, $state.x, $state.y);
        if (d.top != d2.top ||
            d.left != d2.left ||
            d.width != d2.width ||
            d.height != d2.height){
            var tc = $page.tmp_canvas;
            zoom_in($page.full_map, tc, d.left, d.top, d.width, d.height);
            $(tc).css('visibility', 'visible');
        }
    }
}

/** pan_pixel_delta moves the view window, if possible
 *
 * If the view bangs into the edge of the map, it stops moving in that
 * direction.
 *
 * @param dx
 * @param dy
 *
 */
function pan_pixel_delta(dx, dy, dz){
    var z = (1 << $state.zoom);
    var scale_x = $page.x_scale * z;
    var scale_y = $page.y_scale * z;
    var px = dx / scale_x;
    var py = dy / scale_y;
    var x = parseInt($state.x - px);
    var y = parseInt($state.y - py);
    /* because x and y are centre points, they need to be constrained
     * according to the zoom.  But the zoom could be changing
     * simultaneously, so the recalculate that.
     */
    var zoom = $state.zoom + (dz || 0);
    var zz = (1 << zoom);
    var pad_x = $page.range_x / (zz * 2);
    var pad_y = $page.range_y / (zz * 2);
    x = Math.max($page.min_x + pad_x, Math.min($page.max_x - pad_x, x));
    y = Math.max($page.min_y + pad_y, Math.min($page.max_y - pad_y, y));
    set_state({x: parseInt(x), y: parseInt(y), zoom: zoom});
}

function add_known_token_to_ui(token){
    var link;
    var data = $page.token_data[token];
    if (data.count){
        link = $('<div class="token_data">' + token + ' <span class="token_count">' +
                 data.count + "</span></div>");
    }
    else {
        link = $('<div class="no_token_data">' + token +
                 ' <span class="token_count">not mentioned</span></div>');
    }

    link.click(function(){
                   set_state({token: token});
               });
    var div = $("#stored-searches");
    div.append(link);
}
