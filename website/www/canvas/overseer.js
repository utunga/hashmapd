/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/* $const holds constant global state.  Nothing in here should change
 * in the course of the page session.
 *
 * Of course, some of these things *can* be changed right at the
 * beginning, but once data is loaded, they are fixed.
 */
var $const = {
    BASE_DB_URL: ((window.location.hostname == '127.0.0.1') ?
                  'http://127.0.0.1:5984/frontend_dev/_design/user/_view/' :
                  'http://hashmapd.halo.gen.nz:5984/frontend_dev/_design/user/_view/'),
    SQUISH_INTO_CANVAS: false, /*if true, scale X and Y independently, losing map shape */
    USE_JSONP: true,
    ARRAY_FUZZ_CONSTANT: -0.013, /*concetration for array fuzz */
    ARRAY_FUZZ_RADIUS: 18, /*array fuzz goes this far. shouldn't exceed PADDING */
    ARRAY_FUZZ_RADIX: 1.2, /*exponential scaling for hills */
    ARRAY_FUZZ_DENSITY_RADIX: 0, /*0 means linear */
    ARRAY_FUZZ_DENSITY_CONSTANT: -0.007, /*concetration for array fuzz */
    ARRAY_FUZZ_DENSITY_RADIUS: 30, /*array fuzz goes this far */
    ARRAY_FUZZ_TYPED_ARRAY: true, /*whether to use Float32Array or traditional array */
    FUZZ_CONSTANT: -0.015, /*concentration of peaks, negative inverse variance */
    FUZZ_OFFSET: 0.5, /* lift floor by this much (0.5 rounds, more to lengthen tails) */
    FUZZ_PER_POINT: 8, /* a single point generates this much fuzz */
    FUZZ_MAX_RADIUS: 18, /*fuzz never reaches beyond this far */
    FUZZ_MAX_MULTIPLE: 15, /*draw fuzz images for up this many points in one place */
    QUAD_TREE_COORDS: 15,
    COORD_MAX: 1 << 16,   /* exclusive maximum xy coordinates (1 << (QUAD_TREE_COORDS + 1)) */
    COORD_MIN: 0,   /* inclusive minimum xy coordinates. */
    PADDING: 24,    /*padding for the full size map in pixels*/
    width: 800,   /* canvas padded pixel width */
    height: 800,  /* canvas padded pixel height */

    array_fuzz: true,
    HILL_SHADE_FLATNESS: 16.0, /*8 is standard, higher means flatter hills */
    views : {  /* helps in interpreting various views. */
        locations: {},
        token_density: {precision_adjust: 1},
        tokens:{}
    }
};

/* $page holds values that are calculated during a page session and
 * are thereafter constant, including various canvas and DOM
 * references.
 */
var $page = {
    canvas: undefined,  /* a reference to the main canvas gets put here */
    density_canvas: undefined,  /* a smaller scale canvas for subtracting from density overlays*/
    overlays: [],     /*a list of html objects to overlay the main canvas */
    loading: undefined,

    /* convert data coordinates to canvas coordinates. */
    range_x: undefined,
    range_y: undefined,
    x_scale: undefined,
    y_scale: undefined,
    min_x:  undefined,
    min_y:  undefined,
    max_x:  undefined,
    max_y:  undefined,

    tweeters: undefined, /*the parsed user data that makes the main map. */

    trailing_commas_are_GOOD: true
};

/* $waiters is a repository for global $.Deferreds */
var $waiters = {
    map_known: undefined, /*will be a deferred that fires when map scale is known */
    map_drawn: undefined, /*will be a deferred that fires when the landscape is drawn */
    height_map_drawn: undefined,
    hill_fuzz_ready: undefined,
    have_density: undefined,

    trailing_commas_are_GOOD: true
};

/* $state contains values that reflect the manipulable state of page.
 * It has a one-to-one relationship to the URL query string.
 */

var $state = {
    left: 0,    /* left edge of drawn map (0 to COORD_MAX) */
    top: 0,     /* top edge of drawn map */
    zoom: 0,    /* zoom level. 0 is full size, 1 is 1/2, 2 is 1/4, etc */

    labels: false,
    map_resolution: 9,
    density_resolution: 7,

    trailing_commas_are_GOOD: true
};

/* $timestamp is a global timer (once hm_setup is run)*/
var $timestamp;

/** hm_setup does any initialisation that needs to be done once, upon
 * first load.
 *
 * Nothing much happens until the json is loaded.
 */

function hm_setup(){
    $timestamp = get_timer();
    interpret_query();
    $page.loading = loading_screen();
    $page.loading.show("Loading...");

    /* The main map canvas */
    $page.canvas = scaled_canvas();
    $waiters.map_drawn = $.Deferred();
    $waiters.height_map_drawn = $.Deferred();

    /* start downloading the main map */
    $waiters.map_known = get_json('locations', $state.map_resolution, hm_on_data);

    /* if using image based convolution, start making the images */
    if (! $const.array_fuzz){
        $waiters.hill_fuzz_ready = $.Deferred();
        start_fuzz_creation($waiters.hill_fuzz_ready);
    }
    else { /*a non-deferred acts as resolved to $.when() */
        $waiters.hill_fuzz_ready = true;
    }
    $.when($waiters.map_known, $waiters.hill_fuzz_ready).done(make_height_map);
    construct_form();
}

/** hm_draw_map draws the approriate map
 *
 * Nothing much happens until the json is loaded.
 */

function hm_draw_map(){
    interpret_query();
    //$waiters.have_density = get_json('token_density', $state.density_resolution, hm_on_token_density);
    //$waiters.have_density = $.getJSON('tokens-gonna.json', hm_on_token_density);
    //$waiters.have_density = $.getJSON('tokens-check.json', hm_on_token_density);

    if ($state.labels){
        $waiters.have_labels = get_json('tokens', 7, hm_on_labels);
        $.when($waiters.have_labels,
               $waiters.map_drawn).done(paint_labels);
    }

    $.when($waiters.map_known).done(make_density_map);
    $.when($waiters.map_known,
           $waiters.hill_fuzz_ready).done(paint_map);
}


/** get_json fetches data.
 *
 *  @param view is a couchDB view name (e.g. "locations")
 *  @param precision is the desired quadtree precision
 *  @param callback is a callback. It gets the data as first argument.
 *
 *  @return a $.Deferred or $.Deferred-alike object.

 */
function get_json(view, precision, callback){
    /*If the view has non-quadtree data prepended to its key (e.g. a token),
     * then the precision needs to be adjusted accordingly.
     */
    var adjust = $const.views[view].precision_adjust || 0;
    var level = precision + adjust;

    /*inside out compare catches undefined precision, which defaults to deepest level*/
    var group_level = ((precision <= $const.QUAD_TREE_COORDS + adjust) ?
                       "group_level=" + level :
                       "group=true");

    $timestamp("req JSON " + view + "[" + precision + "]");
    var url = $const.BASE_DB_URL + view + '?' + group_level + '&callback=?';
    var d = $.ajax({
                       url: url,
                       dataType: ($const.USE_JSONP) ? 'jsonp': 'json',
                       cache: true, /*not on by default in jsonp mode*/
                       success: function(data){
                           $page.loading.tick();
                           $timestamp("got JSON " + view + "[" + precision + "]");
                           callback(data);
                       }
    });
    return d;
}


/* Start creating fuzz images.  This might take a while and is
 * partially asynchronous.
 *
 *  (It takes 8-40 ms on an i5-540, at time of writing, which beats
 *  JSON loading from local/cached sources.)
 *
 */
function start_fuzz_creation(deferred){
    $timestamp("start make_fuzz");
    $const.hill_fuzz = make_fuzz(
        deferred,
        $const.FUZZ_MAX_MULTIPLE,
        $const.FUZZ_MAX_RADIUS,
        $const.FUZZ_CONSTANT,
        $const.FUZZ_OFFSET,
        $const.FUZZ_PER_POINT);
    $timestamp("end make_fuzz");
}

/** decode_points turns JSON rows into point arrays.
 *
 * The quad tree coordinates are converted to X, Y coordinates.  The
 * final result is an array of arrays, structured thus:
 *
 *  [ [x_coord, y_coord, value],
 *    [x_coord, y_coord, value],
 *  ...]
 *
 * The value is untouched.
 *
 * @param raw  the json data (as parsed by JSON or jquery objects)
 *
 * @return an array of points.
 */

function decode_points(raw){
    var i, j;
    var points = [];
    for (i = 0; i < raw.length; i++){
        var r = raw[i];
        r.special_keys = [];
        var coords = r.key;

        var x = 0;
        var y = 0;
        /*filter out any that aren't numbers and put them in a special place */
        j = 0;
        while (! (typeof(coords[j]) == 'number')){
            r.special_keys.push(coords[j]);
            j++;
        }
        for (; j < coords.length; j++){
            var p = coords[j];
            x = (x << 1) | (p & 1);
            y = (y << 1) | (p >> 1);
        }
        /* add an extra one to put it in the middle of the specified square,
         * rather than the top left corner.
         */
        x = (x << 1) + 1;
        y = (y << 1) + 1;
        /* if these coordinates are less than fully accurate,
         * expand with zeros.
         */
        var n_coords = coords.length - r.special_keys.length;
        x <<= ($const.QUAD_TREE_COORDS - n_coords);
        y <<= ($const.QUAD_TREE_COORDS - n_coords);
        points.push([x, y, r.value, r.special_keys]);
    }
    return points;
}

/** decode_and_filter_points turns JSON rows into point arrays.
 *
 * If you supply <xmin>, <xmax>, <ymin>, or <ymax>, points outside
 * those bounds are excluded.  If any of those are undefined, there is
 * no bound in that direction.
 *
 * If quad tree coordinates are being used, they are converted to X, Y
 * coordinates.  The final result is an array of arrays, structured thus:
 *
 *  [ [x_coord, y_coord, value], [x_coord, y_coord, value], ...]
 *
 * The value is untouched.
 *
 * @param raw  the json data (as parsed by JSON or jsquery objects)
 * @param xmin an exclusive boundary value
 * @param xmax an exclusive boundary value
 * @param ymin an exclusive boundary value
 * @param ymax an exclusive boundary value
 *
 * @return an array of points.
 */

function bound_points(points, xmin, xmax, ymin, ymax){
    /*undefined is equivalent to extreme bounds */
    xmin = (xmin !== undefined) ? xmin : $const.COORD_MIN - 1;
    ymin = (ymin !== undefined) ? ymin : $const.COORD_MIN - 1;
    xmax = (xmax !== undefined) ? xmax : $const.COORD_MAX;
    ymax = (ymax !== undefined) ? ymax : $const.COORD_MAX;
    return points.filter(function(p){
                             return  ((xmin < p[0]) &&
                                      (xmax > p[0]) &&
                                      (ymin < p[1]) &&
                                      (ymax > p[1]));
                         });
}


/** hm_on_data is a callback from hm_draw_map.
 *
 * It coordinates the actual drawing.
 *
 * @param canvas the html5 canvas
 * @param data is parsed but otherwise unprocessed JSON data.
 */

function hm_on_data(data){
    $timestamp("got map data");
    $page.loading.show("Painting");
    var i;
    var width = $const.width;
    var height = $const.height;
    var max_value = 0;
    var points = decode_points(data.rows);
    if ($const.absolute_min_x !== undefined ||
        $const.absolute_max_x !== undefined ||
        $const.absolute_min_y !== undefined ||
        $const.absolute_max_y !== undefined){
        points = bound_points(points,
                              $const.absolute_min_x,
                              $const.absolute_max_x,
                              $const.absolute_min_y,
                              $const.absolute_max_y);
    }
    var max_x = $const.COORD_MIN;
    var max_y = $const.COORD_MIN;
    var min_x = $const.COORD_MAX;
    var min_y = $const.COORD_MAX;
    /*find the coordinate and value ranges */
    for (i = 0; i < points.length; i++){
        var r = points[i];
        max_value = Math.max(r.value, max_value);
        max_x = Math.max(r[0], max_x);
        max_y = Math.max(r[1], max_y);
        min_x = Math.min(r[0], min_x);
        min_y = Math.min(r[1], min_y);
    }
    /*save the discovered extrema for the points, just in case, but
     *the padded values will be more useful for most things.  The
     *discovered extrema are likely to exclude some points if this
     *data is not using the full resolution quadtree.
     */
    $page.point_min_x = min_x;
    $page.point_min_y = min_y;
    $page.point_max_x = max_x;
    $page.point_max_y = max_y;

    var point_range_x = max_x - min_x;
    var pixel_range_x = $const.width - 2 * $const.PADDING;
    var x_scale = pixel_range_x / point_range_x;
    $page.min_x = min_x - $const.PADDING / x_scale;
    $page.max_x = max_x + $const.PADDING / x_scale;

    var point_range_y = max_y - min_y;
    var pixel_range_y = $const.width - 2 * $const.PADDING;
    var y_scale = pixel_range_y / point_range_y;
    $page.min_y = min_y - $const.PADDING / y_scale;
    $page.max_y = max_y - $const.PADDING / y_scale;

    if ($const.SQUISH_INTO_CANVAS){
        $page.x_scale = x_scale;
        $page.y_scale = y_scale;
    }
    else{
        $page.x_scale = Math.min(x_scale, y_scale);
        $page.y_scale = Math.min(x_scale, y_scale);
    }
    $page.tweeters = points;
}

/** make_height_map() depends on  $waiters.hill_fuzz_ready and $waiters.map_known
 */
function make_height_map(){
    $timestamp("start height_map");
    var points = $page.tweeters;
    var canvas = scaled_canvas();
    var ctx = canvas.getContext("2d");
    if ($const.array_fuzz){
        paste_fuzz_array(ctx, points,
                         $const.ARRAY_FUZZ_RADIUS,
                         $const.ARRAY_FUZZ_CONSTANT,
                         $const.ARRAY_FUZZ_RADIX
                        );
    }
    else{
        paste_fuzz(ctx, points, $page.hill_fuzz);
    }
    $page.height_canvas = canvas;
    $timestamp("end height_map");

    $waiters.height_map_drawn.resolve();
}


function get_zoom_pixel_bounds(zoom, left, top){
    var w = $page.canvas.width;
    var h = $page.canvas.height;
    var zw = w / (1 << zoom);
    var zh = h / (1 << zoom);
    var x = Math.min(left * w / $const.COORD_MAX, w - zw);
    var y = Math.min(top * h / $const.COORD_MAX, h - zh);
    return {
        x: x,
        y: y,
        width: zw,
        height: zh
    };
}

function get_zoom_point_bounds(zoom, left, top){
    var size = COORD_MAX / (1 << zoom);
    var outside = $const.COORD_MAX - size;
    var min_x = Math.min(left, outside);
    var min_y = Math.min(top, outside);
    return {
        min_x: min_x,
        min_y: min_y,
        max_x: min_x + size,
        max_y: min_y + size
    };
}


function paint_map(){
    $timestamp("start paint_map");
    var points = $page.tweeters;
    var height_map;
    if ($state.zoom){
        height_map = scaled_canvas();
        var height_ctx = height_map.getContext("2d");
        var w = height_map.width;
        var h = height_map.height;
        var d = get_zoom_pixel_bounds($state.zoom, $state.left, $state.top);
        height_ctx.drawImage($page.height_canvas, d.x, d.y, d.width, d.height,
                             0, 0, height_map.width, height_map.height);
    }
    else {
        height_map = $page.height_canvas;
        var height_ctx = height_map.getContext("2d");
    }
    var canvas = $page.canvas;
    var ctx = canvas.getContext("2d");
    $timestamp("start hillshading");
    hillshading(height_ctx, ctx, 1 / ($state.zoom + 1), Math.PI * 1 / 4, Math.PI / 4);
    $timestamp("end paint_map");
    $waiters.map_drawn.resolve();
    $page.loading.done();
}


function make_density_map(){
    var canvas = scaled_canvas(0.25);
    var ctx = canvas.getContext("2d");
    $.when($waiters.map_known).done(function()
                       {
                          paint_density_array(ctx, $page.tweeters);
                       });
    $page.density_canvas = canvas;
}

function hm_on_token_density(data){
    log("in hm_on_token_density");
    var points = decode_points(data.rows);
    var token_canvas = scaled_canvas(0.25);
    var token_ctx = token_canvas.getContext("2d");
    $.when($waiters.map_known).done(function(){
                           points = bound_points(points,
                                                 $page.min_x, $page.max_x,
                                                 $page.min_y, $page.max_y);
                           $timestamp("pre density map");

                           if ($const.array_fuzz){
                               paint_density_array(token_ctx, points);
                           }
                           else{
                               paste_fuzz(token_ctx, points, $page.hill_fuzz);
                           }
                           $timestamp("applying density map");
                           var token_canvas2 = apply_density_map(token_ctx);
                           $timestamp("post density map");
                           $page.overlays.push(token_canvas2);
                           $(token_canvas2).addClass("overlay").offset(
                               $($page.canvas).offset());
                       });
}


/*don't do too much until the drawing is done.*/

function hm_on_labels(data){
    var points = decode_points(data.rows);
    /*XXX depends on map_known */
    points = bound_points(points,
                          $page.min_x, $page.max_x,
                          $page.min_y, $page.max_y);
    var max_freq = 0;
    for (var i = 0; i < points.length; i++){
        var freq = points[i][2][0][1];
        max_freq = Math.max(freq, max_freq);
    }
    var scale = 14 / (max_freq * max_freq);

    $state.labels = {
        points: points,
        max_freq: max_freq,
        scale: scale
    };
}

function paint_labels(){
    var points = $page.labels.points;
    var scale = $page.labels.scale;
    var ctx = $page.canvas.getContext("2d");
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        var x = (p[0] - $page.min_x) * $page.x_scale;
        var y = (p[1] - $page.min_y) * $page.y_scale;
        var text = p[2][0][0];
        var n = p[2][0][1];
        var size = n * n * scale;
        add_label(ctx, text, x, y, size, "#000", "#fff");
    }
}

/** interpret_query puts any well specified query parameters into $state
 *
 * This feels completely dodgy to a server side programmer, but is OK
 * on the client side.  They can only mangle their own browsers.
 */

function interpret_query(dest, query){
    if (dest === undefined){
        dest = $state;
    }
    if (query === undefined || typeof(query) == 'string'){
        query = get_query(query);
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

function get_query(query){
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

    form.append(param + '<button name="go" value="go">');
    form.submit(submit);
}

function set_state(data){
    var q;
    if (typeof(data) == 'string'){
        q = data;
    }
    else{
        q = $.param(data);
    }
    alert(q);
    interpret_query($state, q);
    //window.location.search = "?" + q;
    var h = window.history;
    var loc = window.location;
    var url = loc.href.split("?", 1)[0] + "?" + q;
    h.pushState($state, "Hashmapd", url);
}