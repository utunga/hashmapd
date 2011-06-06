/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/* $hm holds global state.  Capitalised names are assumed to be
 * constant (unnecessarily in some cases).
 *
 * Undefined properties are of course only included here by way of
 * documention.
 */
var $hm = {
    SQUISH_INTO_CANVAS: false, /*if true, scale X and Y independently, losing map shape */
    DATA_URL: 'locations-9.json',
    TOKEN_DENSITY_URL: 'token_density-8.json',
    LABELS_URL: 'tokens-7.json',
    //DATA_URL: 'http://hashmapd.couchone.com/frontend_dev/_design/user/_view/xy_coords?group=true',
    PADDING: 20,    /*padding for the image as a whole. */
    ARRAY_FUZZ_CONSTANT: -0.02, /*concetration for array fuzz */
    ARRAY_FUZZ_RADIUS: 16, /*array fuzz goes this far. shouldn't exceed PADDING */
    ARRAY_FUZZ_RADIX: 1.25, /*exponential scaling for hills */
    ARRAY_FUZZ_DENSITY_RADIX: 0, /*0 means linear */
    ARRAY_FUZZ_DENSITY_CONSTANT: -0.017, /*concetration for array fuzz */
    ARRAY_FUZZ_DENSITY_RADIUS: 16, /*array fuzz goes this far */
    ARRAY_FUZZ_TYPED_ARRAY: true, /*whether to use Float32Array or traditional array */
    FUZZ_CONSTANT: -0.015, /*concentration of peaks, negative inverse variance */
    FUZZ_OFFSET: 0.5, /* lift floor by this much (0.5 rounds, more to lengthen tails) */
    FUZZ_PER_POINT: 8, /* a single point generates this much fuzz */
    FUZZ_MAX_RADIUS: 18, /*fuzz never reaches beyond this far */
    FUZZ_MAX_MULTIPLE: 15, /*draw fuzz images for up this many points in one place */
    USING_QUAD_TREE: true,
    QUAD_TREE_COORDS: 15,
    map_known: undefined, /*will be a deferred that fires when map scale is known */
    map_drawn: undefined, /*will be a deferredthat fires when the landscape is drawn */
    canvas: undefined,  /* a reference to the main canvas gets put here */
    width: 800,   /* canvas *unpadded* pixel width */
    height: 600,  /* canvas *unpadded* pixel height */

    /* convert data coordinates to canvas coordinates. */
    range_x: undefined,
    range_y: undefined,
    x_scale: undefined,
    y_scale: undefined,
    min_x:  undefined,
    min_y:  undefined,
    max_x:  undefined,
    max_y:  undefined,
    overlays: [],     /*a list of html objects to overlay the main canvas */
    array_fuzz: true,
    labels: false,


    trailing_commas_are_GOOD: true
};

/** hm_draw_map is the main entrance point.
 *
 * Nothing much happens until the json is loaded.
 */

function hm_draw_map(){
    $hm.timer = get_timer();
    interpret_query();

    $hm.canvas = fullsize_canvas();
    $hm.map_known = $.Deferred();
    $hm.map_drawn = $.Deferred();
    $hm.have_labels = $.Deferred();
    $hm.have_density = $.Deferred();

    if (! $hm.array_fuzz)
        start_fuzz_creation();

    $.getJSON($hm.DATA_URL, function(data){
                  hm_on_data(data);
              });

    $.getJSON($hm.TOKEN_DENSITY_URL, function(data){
                  hm_on_token_density(data);
              });
    if ($hm.labels){
        $.getJSON($hm.LABELS_URL, function(data){
                      hm_on_labels(data);
                  });
    }
    $hm.map_known.then(paint_map);
    $hm.have_labels.then(paint_labels);
    $hm.have_density.then(paint_density_map);
}

/* Start creating fuzz images.  This might take a while and is
 * partially asynchronous.
 *
 *  (It takes 8-40 ms on an i5-540, at time of writing, which beats
 *  JSON loading from local/cached sources.)
 *
 */
function start_fuzz_creation(){
    $hm.timer.checkpoint("start make_fuzz");
    $hm.hill_fuzz = make_fuzz(
        $.Deferred(),
        $hm.FUZZ_MAX_MULTIPLE,
        $hm.FUZZ_MAX_RADIUS,
        $hm.FUZZ_CONSTANT,
        $hm.FUZZ_OFFSET,
        $hm.FUZZ_PER_POINT);
    $hm.timer.checkpoint("end make_fuzz");
}

function paint_map(){
    if ($hm.array_fuzz){
        _paint_map();
    }
    else{
        $hm.hill_fuzz.ready.then(_paint_map);
    }
}

function paint_labels(){
    $hm.map_drawn.then(_paint_labels);
}

function paint_density_map(){
    $hm.map_drawn.then(
        function(){
            if ($hm.array_fuzz){
                _paint_density_map();
            }
            else{
                $hm.hill_fuzz.ready.then(_paint_density_map);
            }
        }
    );
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
        /* if these coordinates are less than fully accurate,
         * expand with zeros.
         */
        var n_coords = coords.length - r.special_keys.length;
        x <<= ($hm.QUAD_TREE_COORDS - n_coords);
        y <<= ($hm.QUAD_TREE_COORDS - n_coords);
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
    /*undefined is equivalent to +/- inf */
    xmin = (xmin !== undefined) ? xmin : -1e999;
    ymin = (ymin !== undefined) ? ymin : -1e999;
    xmax = (xmax !== undefined) ? xmax :  1e999;
    ymax = (ymax !== undefined) ? ymax :  1e999;
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
    $hm.timer.checkpoint("got map data");
    var i;
    var width = $hm.width;
    var height = $hm.height;
    var max_value = 0;
    var max_x = -1e999;
    var max_y = -1e999;
    var min_x =  1e999;
    var min_y =  1e999;
    var points = decode_points(data.rows);
    /*find the coordinate and value ranges */
    for (i = 0; i < points.length; i++){
        var r = points[i];
        max_value = Math.max(r.value, max_value);
        max_x = Math.max(r[0], max_x);
        max_y = Math.max(r[1], max_y);
        min_x = Math.min(r[0], min_x);
        min_y = Math.min(r[1], min_y);
    }
    $hm.tweeters = points;
    $hm.range_x = max_x - min_x;
    $hm.range_y = max_y - min_y;
    var x_scale = width / $hm.range_x;
    var y_scale = height / $hm.range_y;
    if ($hm.SQUISH_INTO_SHAPE){
        $hm.x_scale = x_scale;
        $hm.y_scale = y_scale;
    }
    else{
        $hm.x_scale = Math.min(x_scale, y_scale);
        $hm.y_scale = Math.min(x_scale, y_scale);
        /*XXX range_x, range_y too?*/
    }
    $hm.min_x = min_x;
    $hm.min_y = min_y;
    $hm.max_x = max_x;
    $hm.max_y = max_y;
    $hm.map_known.resolve();
}

/** _paint_map() depends on  $hm.hill_fuzz.ready and $hm.map_known
 */

function _paint_map(){
    var points = $hm.tweeters;
    var canvas = $hm.canvas;
    var ctx = canvas.getContext("2d");
    var fuzz_canvas = fullsize_canvas();
    var fuzz_ctx = fuzz_canvas.getContext("2d");
    $hm.timer.checkpoint("start paste_fuzz");
    if ($hm.array_fuzz){
        paste_fuzz_array(fuzz_ctx, points,
                         $hm.ARRAY_FUZZ_RADIUS,
                         $hm.ARRAY_FUZZ_CONSTANT,
                         $hm.ARRAY_FUZZ_RADIX
                        );
    }
    else{
        paste_fuzz(fuzz_ctx, points, $hm.hill_fuzz);
    }
    $hm.timer.checkpoint("end paste_fuzz");
    $hm.timer.checkpoint("start hillshading");
    hillshading(fuzz_ctx, ctx, 1, Math.PI * 1 / 4, Math.PI / 4);
    $hm.timer.checkpoint("end hillshading");
    $hm.map_drawn.resolve();
}


function wait_for_flag(flag, func){
    if ($hm[flag]){
        func();
    }
    else {
        window.setTimeout(wait_for_flag, 100, flag, func);
    }
}


function hm_on_token_density(data){
    log("in hm_on_token_density");
    $hm.timer.doing_tokens = Date.now();
    var i;
    var points = decode_points(data.rows);
    points = bound_points(points,
                          $hm.min_x, $hm.max_x,
                          $hm.min_y, $hm.max_y);

    var token_canvas = fullsize_canvas();
    var token_ctx = token_canvas.getContext("2d");

    var max_freq = 0;
    for (var i = 0; i < points.length; i++){
        var freq = points[i][2];
        max_freq = Math.max(freq, max_freq);
    }

    if ($hm.array_fuzz){
        $hm.map_known.then(
            function _paste_fuzz (){
                $hm.timer.checkpoint("pre density map");
                //hidden_fill(token_ctx, "#ff0");
                token_ctx.fillStyle = "#ff0";
                token_ctx.fillRect(0, 0, token_canvas.width, token_canvas.height);
                paste_fuzz_array(token_ctx, points,
                                 $hm.ARRAY_FUZZ_DENSITY_RADIUS,
                                 $hm.ARRAY_FUZZ_DENSITY_CONSTANT,
                                 $hm.ARRAY_FUZZ_DENSITY_RADIX
                                );
                $hm.timer.checkpoint("post density map");
                }
        );
    }
    else{
        $hm.hill_fuzz.ready.then(
            function(){
                paste_fuzz(token_ctx, points, $hm.hill_fuzz);
            }
        );
    }
    $hm.overlays.push(token_canvas);
    $hm.have_density.resolve();
}


function _paint_density_map(){
    $($hm.overlays[0]).addClass("overlay").offset($($hm.canvas).offset());
}



/*don't do too much until the drawing is done.*/

function hm_on_labels(data){
    var points = decode_points(data.rows);
    /*XXX depends on map_known */
    points = bound_points(points,
                          $hm.min_x, $hm.max_x,
                          $hm.min_y, $hm.max_y);
    var max_freq = 0;
    for (var i = 0; i < points.length; i++){
        var freq = points[i][2][0][1];
        max_freq = Math.max(freq, max_freq);
    }
    var scale = 14 / (max_freq * max_freq);

    $hm.labels = {
        points: points,
        max_freq: max_freq,
        scale: scale
    };
    $hm.have_labels.resolve();
}

function _paint_labels(){
    var points = $hm.labels.points;
    var scale = $hm.labels.scale;
    var ctx = $hm.canvas.getContext("2d");
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        var x = $hm.PADDING + (p[0] - $hm.min_x) * $hm.x_scale;
        var y = $hm.PADDING + (p[1] - $hm.min_y) * $hm.y_scale;
        var text = p[2][0][0];
        var n = p[2][0][1];
        var size = n * n * scale;
        add_label(ctx, text, x, y, size, "#000", "#fff");
    }
}

/** interpret_query puts any well specified query parameters into $hm
 *
 * This feels completely dodgy to a server side programmer, but is OK
 * on the client side.  They can only mangle their own browsers.
 */

function interpret_query(){
    var query = get_query();
    for (var param in query){
        if (param in $hm){
            var v = query[param];
            var existing = $hm[param];
            switch(typeof(existing)){
            case "number":
                v = parseFloat(v);
                if (! isNaN(v)){
                    $hm[param] = v;
                }
                break;
            case "boolean":
                v = v.toLowerCase();
                $hm[param] = (!(v == "0" ||
                                v == "false" ||
                                v == "no" ||
                                v == ""));
                break;
            case "string":
                $hm[param] = v;
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

function get_query(){
    var query = window.location.search.substring(1);
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
