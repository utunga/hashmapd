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
    DEBUG: (window.location.pathname.substr(-10) == 'debug.html'),
    BASE_DB_URL: 'http://couch.hashmapd.com/fd/',
    //BASE_DB_URL: 'http://127.0.0.1:5984/frontend_dev/_design/user/_view/',
    SQUISH_INTO_CANVAS: false, /*if true, scale X and Y independently, losing map shape */
    USE_JSONP: true,
    FPS: 20, /*how often is the animation tick (frames per second)*/
    ARRAY_FUZZ_SCALE: 255, /*highest peak is this high*/
    ARRAY_FUZZ_LUT_LENGTH: 2000, /*granularity of height conversion LUT */
    ARRAY_FUZZ_CONSTANT: -0.013, /*concentration for array fuzz */
    ARRAY_FUZZ_THRESHOLD: 0.005, /*array fuzz gets this faint */
    /* *_SCALE_ARGS, out of ['linear'], ['clipped_gaussian', low, high], ['base', base] */
    ARRAY_FUZZ_SCALE_ARGS: ['clipped_gaussian', -3.5, -0.4],
    ARRAY_FUZZ_DENSITY_SCALE_ARGS: ['linear'],
    ARRAY_FUZZ_DENSITY_CONSTANT: -0.007, /*concentration for array fuzz */
    ARRAY_FUZZ_DENSITY_THRESHOLD: 0.005, /*density fuzz gets this faint */
    ARRAY_FUZZ_TYPED_ARRAY: true, /*whether to use Float32Array or traditional array */
    REDRAW_HEIGHT_MAP: true, /*whether to redraw the height map on zoom */
    MAP_RESOLUTION: 9,       /*initial requested resolution for overall map*/
    DENSITY_RESOLUTION: 7,   /*initial requested resolution for density maps*/
    DENSITY_MAP_STYLE: 'grey_outside',
    //DENSITY_MAP_STYLE: 'colour_cycle',

    KEY_CAPTURE: false, /* use keyboard events to scroll and zoom */

    DENSITY_MAP_ZOOM_DETAIL: true,

    QUAD_TREE_COORDS: 15,
    COORD_MAX: 1 << 16,   /* exclusive maximum xy coordinates (1 << (QUAD_TREE_COORDS + 1)) */
    COORD_MIN: 0,   /* inclusive minimum xy coordinates. */
    /*PADDING should be both:
     * - bigger than the fuzz radius used to construct the map (so edge is sea).
     *
     * - big enough that PADDING/{width,height} is greater than the
     *   inverse minimum resolution. That is, > 800/128 if 7 quad tree
     *   coordinates are used with an 800x800 canvas.
     */
    PADDING: 24,    /*padding for the full size map in pixels*/
    width: 800,   /* canvas padded pixel width */
    height: 800,  /* canvas padded pixel height */

    HILL_LUT_CENTRE: 300,
    HILL_SHADE_FLATNESS: 16.0, /*8 is standard, higher means flatter hills */
    views : {  /* helps in interpreting various views. */
        locations: {},
        token_density: {precision_adjust: 1},
        tokens:{}
    },
    DENSITY_OPS: {
            '+': density_add,
            '*': density_mul,
            '-': density_sub,
            '^': density_diff
        }
};

/* $page holds values that are calculated during a page session and
 * are thereafter constant, including various canvas and DOM
 * references.
 */
var $page = {
    canvases: {},       /*Named canvases go in here (see named_canvas()) */
    canvas: undefined,  /* a reference to the main canvas gets put here */
    full_map: undefined, /* will be the full unzoomed */
    loading: undefined,
    labels: undefined,  /* JSON derived structure describing labels */
    height_canvas: undefined, /*a height map canvas */
    max_height: undefined, /* maximum value of array fuzz */

    /* convert data coordinates to canvas coordinates. */
    range_x: undefined,
    range_y: undefined,
    x_scale: undefined,
    y_scale: undefined,
    min_x:  undefined,
    min_y:  undefined,
    max_x:  undefined,
    max_y:  undefined,
    hillshade_luts: {},
    /*token_data is a cache of token density data, with a list of points for each
     * known token. Like so:
     * {'LOL': [[x, y, value, precision], [x, y, value, precision], ...],
     *  'Orange': [...],
     * ...}
     *  */
    token_data: {},
    tweeters: undefined, /*the parsed user data that makes the main map. */

    trailing_commas_are_GOOD: true
};

/* $waiters is a repository for global $.Deferreds */
var $waiters = {
    map_known: undefined, /*will be a deferred that fires when map scale is known */
    map_drawn: undefined, /*will be a deferred that fires when $page.canvas is ready */
    full_map_drawn: undefined, /*will fire when a canonical unzoomed map is at $page.full_map */
    height_map_drawn: undefined,
    have_density: undefined,
    density_drawn: undefined,

    trailing_commas_are_GOOD: true
};

/* $state contains values that reflect the manipulable state of page.
 * It has a one-to-one relationship to the URL query string.
 */

var $state = {
    x: 0,    /* centre of drawn map (0 to COORD_MAX) */
    y: 0,
    zoom: 0,    /* zoom level. 0 is full size, 1 is 1/2, 2 is 1/4, etc */
    token: '',

    labels: true
};

/* $timestamp is a global timer (once hm_setup is run)*/
var $timestamp;

/** hm_setup does any initialisation that needs to be done once, upon
 * first load.
 *
 * Nothing much happens until the json is loaded.
 */

function hm_setup(){
    /*load matching query parameters into $const, just this once. */
    interpret_query($const);
    $timestamp = get_timer();
    /*now load them into $state. This happens regularly in hm_draw_map */
    interpret_query($state);

    $page.loading = loading_screen();
    $page.loading.show("Loading...");

    /* The main map canvas */
    $page.canvas = named_canvas('main');
    $("#map-div").append($page.canvas);
    $page.tmp_canvas = overlay_canvas('temp', true);
    overlay_canvas("density_overlay");

    $waiters.full_map_drawn = $.Deferred();
    $waiters.height_map_drawn = $.Deferred();

    /* start downloading the main map */
    $waiters.map_known = get_json('locations', $const.MAP_RESOLUTION, hm_on_data);

    $.when($waiters.map_known).done(make_height_map, make_full_map);
    if ($const.DEBUG){
        construct_msgbox();
        construct_form($state, 'state-form', function() {
                           var q = this.serialize();
                           set_state(q);
                           return false;
                       });
        construct_form($const, 'const-form');
    }
    construct_ui();
    enable_drag();
    /*start the animation loop when the main map is done */
    $.when($waiters.full_map_drawn).done(
        function(){
            $page.ticker = window.setInterval(hm_tick, 1000/ $const.FPS);
        });
    /*
    if ($state.token){
        $waiters.have_density = parse_density_query($state.token);
    }
     */
}

/** hm_draw_map draws the approriate map
 *
 * Nothing much happens until the json is loaded.
 */

function hm_draw_map(){
    $waiters.map_drawn = $.Deferred();
    $timestamp("start hm_draw_map", true);
    interpret_query($state);
    set_ui($state);
    if ($state.token){
        $waiters.have_density = parse_density_query($state.token);
    }
    temp_view();
    /*have a short break here to allow the temp view to show */
    window.setTimeout(hm_draw_map2, 1);
}

function hm_draw_map2(){
    if ($state.labels){
        $waiters.have_labels = get_label_json(hm_on_labels);
        log($waiters.have_labels);
        $.when($waiters.have_labels,
                $waiters.map_known).done(paint_labels);
    }

    $.when($waiters.map_known).done(paint_map);
    $.when($waiters.map_drawn,
           $waiters.have_density).done(hide_temp_view);
}



/** get_json fetches data.
 *
 *  @param view is a couchDB view name (e.g. "locations")
 *  @param precision is the desired quadtree precision
 *  @param callback is a callback. It gets the data as first argument.
 *  @param start exclude values before this
 *  @param end exclude values after this (use "[x,{}]" include x*).
 *  @param context data to be passed to the callback.
 *
 *  @return a $.Deferred or $.Deferred-alike object.

 */
function get_json(view, precision, callbacks, start, end, context){
    /*If the view has non-quadtree data prepended to its key (e.g. a token),
     * then the precision needs to be adjusted accordingly.
     */
    $timestamp("req JSON " + view + "[" + precision + "]");
    var adjust = $const.views[view].precision_adjust || 0;
    var level = precision + adjust;

    var args = {stale: 'ok'};

    /*inside out compare catches undefined precision, which defaults to deepest level*/
    if (precision <= $const.QUAD_TREE_COORDS + adjust)
        args.group_level = level;
    else
        args.group = "true";
    if (start) args.startkey = start;
    if (end) args.endkey = end;

    var jsonp_callback = ($const.USE_JSONP) ? '&callback=?': '';
    var url = ($const.BASE_DB_URL + view + '?' + $.param(args)) + jsonp_callback;
    var d = $.ajax({
                       url: url,
                       dataType: ($const.USE_JSONP) ? 'jsonp': 'json',
                       cache: true, /*not on by default in jsonp mode*/
                       context: context,/*will be 'this' in callbacks */
                       success: function(data){
                           log(context, this);
                           $page.loading.tick();
                           $timestamp("got JSON " + view + "[" + precision + "]");
                           /* Actually jquery can do this list of callbacks thing.
                            *nevermind. */
                           if (typeof(callbacks) == 'function'){
                               callbacks = [callbacks];
                           }
                           for (var i = 0; i < callbacks.length; i++){
                               callbacks[i](data, context);
                           }
                       }
    });
    return d;
}

function get_token_json(token, precision, callbacks){
    var startkey = '["' + token + '"]';
    var endkey = '["' + token + '",{}]';
    return get_json('token_density', precision, callbacks, startkey, endkey,
                    {token: token});
};

function get_label_json(callback){
    log("getting labels");
    var d = $.getJSON("labels.json", callback);
    return d;
};

/** If a token is not cached locally, fetch it.
 *
 * @param token a token (single word, usually capitalised)
 * @return a deferred or true (which works as expected for $.when()).
 */
function maybe_get_token_json(token){
    if (! (token in $page.token_data)){
        return get_token_json(token,
                              $const.DENSITY_RESOLUTION,
                              hm_on_token_density);
    }
    return true;
}

function parse_density_query(query){
    var tokens = query.split(/\s/, 4);
    var op, func, args;
    var deferred;
    if (tokens.length == 3 && tokens[1] in $const.DENSITY_OPS){
        /* token arithmetic */
        func = paint_density_duo;
        op = $const.DENSITY_OPS[tokens[1]];
        var d1 = maybe_get_token_json(tokens[0]);
        var d2 = maybe_get_token_json(tokens[2]);
        deferred = $.when(d1, d2);
        args = [tokens[0], tokens[2], op];
    }
    else {
        /* the simple case of one token (possibly with ignored garbage) */
        deferred = maybe_get_token_json(tokens[0]);
        func = paint_token_density;
        args = tokens;
    }
    $.when(args, $waiters.map_drawn, $waiters.full_map_drawn, deferred).done(func);
    return deferred;
}



/** encode_point turns an x, y pair into a quad-tree coordinate.
*/
function encode_point(x, y){
    var qt = [];
    /*last coord is always false */
    x >>= 1;
    y >>= 1;
    for (var i = $const.QUAD_TREE_COORDS - 1; i >= 0; i--){
        qt[i] = (y & 1) * 2 + (x & 1);
        y >>= 1;
        x >>= 1;
    }
    return qt;
}


/** decode_points turns JSON rows into point arrays.
 *
 * The quad tree coordinates are converted to X, Y coordinates.  The
 * final result is an array of arrays, structured thus:
 *
 *  [ [x_coord, y_coord, value, precision, extra],
 *    [x_coord, y_coord, value, precision, extra],
 *  ...]
 *
 * where precision indicates the number of quadtree coordinates found,
 * and extra is any string data that preceded the quadtree coordinates
 * (usually a token name).
 *
 * The value is untouched.  In practice it is usually an integer
 * count, but could be anything encodable in json.
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
        var precision = coords.length - r.special_keys.length;
        x <<= ($const.QUAD_TREE_COORDS - precision);
        y <<= ($const.QUAD_TREE_COORDS - precision);
        points.push([x, y, r.value, precision, r.special_keys]);
    }
    return points;
}

/** bound_points filters a list of points, rejecting those out of bounds
 *
 * If you supply <xmin>, <xmax>, <ymin>, or <ymax>, points outside
 * those bounds are excluded.  If any of those are undefined, there is
 * no bound in that direction.
 *
 * The value is untouched.
 *
 * @param points the points to filter
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
    /* $const.absolute_{min,max}_{x,y} can be used to restrict the
     * overall map to some region.  But you should really try to do
     * that on the server.
     */
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
    /* add some padding to the discovered extrema.  This has two
     * purposes:
     * 1. The edge of the map will be sea.
     * 2. Some points may lie outside the discovered points at higher
     * levels of precision.
     */
    var point_range_x = max_x - min_x;
    var pixel_range_x = $const.width - 2 * $const.PADDING;
    var x_scale = pixel_range_x / point_range_x;
    $page.min_x = min_x - $const.PADDING / x_scale;

    var point_range_y = max_y - min_y;
    var pixel_range_y = $const.height - 2 * $const.PADDING;
    var y_scale = pixel_range_y / point_range_y;
    $page.min_y = min_y - $const.PADDING / y_scale;

    if ($const.SQUISH_INTO_CANVAS){
        $page.x_scale = x_scale;
        $page.y_scale = y_scale;
    }
    else{
        $page.x_scale = Math.min(x_scale, y_scale);
        $page.y_scale = Math.min(x_scale, y_scale);
    }
    $page.max_x = $page.min_x + $const.width / $page.x_scale;
    $page.max_y = $page.min_y + $const.width / $page.y_scale;
    $page.range_x = $page.max_x - $page.min_x;
    $page.range_y = $page.max_y - $page.min_y;

    $page.tweeters = points;
}

/** make_height_map() depends on  $waiters.map_known
 */
function make_height_map(){
    $timestamp("start height_map");
    var points = $page.tweeters;
    var canvas = named_canvas("height_map", "rgba(255,0,0, 1)", 1);
    var ctx = canvas.getContext("2d");

    var map = make_fuzz_array(points,
                              $const.ARRAY_FUZZ_CONSTANT,
                              $const.ARRAY_FUZZ_THRESHOLD,
                              $const.width, $const.height,
                              $page.min_x, $page.min_y,
                              $page.x_scale, $page.y_scale);
    $page.max_height = paste_fuzz_array(ctx, map,
                                        $const.ARRAY_FUZZ_SCALE_ARGS);

    $page.height_canvas = canvas;
    $timestamp("end height_map");

    $waiters.height_map_drawn.resolve();
}

/** make_full_map draws the full map on an auxillary canvas.
 *
 * It is run when the page first loads, and puts the map in
 * $page.full_map.  The map is used for temporary images while the
 * main canvas is being redrawn. */
function make_full_map(){
    $timestamp("start make_full_map");
    var points = $page.tweeters;
    var canvas = named_canvas("full_map");
    var ctx = canvas.getContext("2d");
    var height_map = $page.height_canvas;
    var height_ctx = height_map.getContext("2d");
    hillshading(height_ctx, ctx, 1, Math.PI * 1 / 4, Math.PI / 4);
    $page.full_map = canvas;
    $waiters.full_map_drawn.resolve();
    $timestamp("end make_full_map");
}

function get_pixel_coords(x, y, state){
    state = state || $state;
    var z = get_zoom_point_bounds(state.zoom, state.x, state.y);
    return {
        x: (x - z.min_x) * z.x_scale,
        y: (y - z.min_y) * z.y_scale
    };
}


function get_zoom_pixel_bounds(zoom, centre_x, centre_y, width, height){
    var z = get_zoom_point_bounds(zoom, centre_x, centre_y, width, height);
    var scale = 1.0 / (1 << zoom);
    var x_scale = z.x_scale * scale;
    var y_scale = z.y_scale * scale;

    return {
        left: (z.min_x - $page.min_x) * x_scale,
        top: (z.min_y - $page.min_y) * y_scale,
        width: z.width * x_scale,
        height: z.height * y_scale
    };
}

function get_zoom_point_bounds(zoom, centre_x, centre_y, width, height){
    if (width === undefined)
        width = $const.width;
    if (height === undefined)
        height = $const.height;

    var scale = 1.0 / (1 << zoom);
    var size_x = $page.range_x * scale;
    var size_y = $page.range_y * scale;
    var out_x = $page.range_x - size_x;
    var out_y = $page.range_y - size_y;
    var left = Math.max($page.min_x, centre_x - size_x / 2);
    var top = Math.max($page.min_y, centre_y - size_y / 2);
    var min_x = Math.min(left, $page.min_x + out_x);
    var min_y = Math.min(top, $page.min_y + out_y);
    var x_scale = width / size_x;
    var y_scale = height / size_y;

    var z = {
        min_x: min_x,
        min_y: min_y,
        max_x: min_x + size_x,
        max_y: min_y + size_y,
        x_scale: x_scale,
        y_scale: y_scale,
        width: size_x,
        height: size_y
    };
    return z;
}


function paint_map(){
    $timestamp("start paint_map");
    var points = $page.tweeters;
    var height_map;
    var zoom = $state.zoom;
    if (zoom){
        var scale = 1 << zoom;
        height_map = named_canvas("zoomed_height_map", true, 1);
        var height_ctx = height_map.getContext("2d");
        if ($const.REDRAW_HEIGHT_MAP){
            var height = zoomed_paint(height_ctx, points, zoom,
                                  $const.ARRAY_FUZZ_CONSTANT,
                                  $const.ARRAY_FUZZ_THRESHOLD,
                                  $const.ARRAY_FUZZ_SCALE_ARGS,
                                  $page.max_height);
        }
        else {
            var d = get_zoom_pixel_bounds(zoom, $state.x, $state.y);
            zoom_in($page.height_canvas, height_ctx, d.left, d.top, d.width, d.height);
        }
    }
    else {
        height_map = $page.height_canvas;
        var height_ctx = height_map.getContext("2d");
    }
    var canvas = $page.canvas;
    var ctx = canvas.getContext("2d");
    $timestamp("start hillshading");
    hillshading(height_ctx, ctx, 1 / (zoom + 1), Math.PI * 1 / 4, Math.PI / 4);
    $timestamp("end paint_map");
    $waiters.map_drawn.resolve();
    $page.loading.done();
}


function hm_on_token_density(data, req){
    log("in hm_on_token_density");
    var points = decode_points(data.rows);
    var i, p;
    var token = req.token;
    var cache = $page.token_data;
    if (token in cache){
        /* this is presumably a request for more precision, which we
         * don't yet handle.
         *
         * The thing to do is, for each received point, work out which
         * existing point it is a subset of, and replace that.
         */
        dump_object(req);
        dump_object(cache);
        //dump_object(cache[token]);
        log("got", token, "but it is already in cache, so I WILL IGNORE IT.");
    }
    else {
        log("starting density cache for", token);
        var count = 0;
        for (i = 0; i < points.length; i++){
            p = points[i];
            p.pop();
            count += p[2];
        }
        cache[token] = {
            count: count,
            points: points
        };
    }
}



/*don't do too much until the drawing is done.*/

function hm_on_labels(data){
    log("got labels");
    var points = decode_points(data.rows);
    /*XXX depends on map_known */
    points = bound_points(points,
                          $page.min_x, $page.max_x,
                          $page.min_y, $page.max_y);
    var max_size = 0;;
    for (i = 0; i < points.length; i++){
        var p = points[i];
        var size = points[i][2];
        max_size = Math.max(size, max_size);
    }
    $page.labels = {
        points: points,
        max_size: max_size
    };
}

function paint_labels(){
    log("painting labels");
    var points = $page.labels.points;
    var canvas = overlay_canvas("labels", undefined, true);
    var ctx = canvas.getContext("2d");
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        log(p);
        var d = get_pixel_coords(p[0], p[1]);
        if (d.x < 0 || d.x >= $const.width ||
            d.y < 0 || d.y >= $const.height){
            continue;
        }
        var text = p[4];
        var n = p[2];
        var size = n / 50 * (1 + $state.zoom);
        add_label(ctx, text, d.x, d.y, size, "#000", "#fff");
    }
}


function temp_view(){
    $page.transition = true;
    $.when($waiters.full_map_drawn).done(
        function(){
            if ($page.transition){
                var tc = $page.tmp_canvas;
                var d = get_zoom_pixel_bounds($state.zoom, $state.x, $state.y);
                zoom_in($page.full_map, tc, d.left, d.top, d.width, d.height);
                overlay(tc);
            }
        }
    );
}

function hide_temp_view(){
    $page.transition = false;
    $($page.tmp_canvas).css('visibility', 'hidden');
}
