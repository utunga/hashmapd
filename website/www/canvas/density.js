/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/** is_token_bad returns a true value if a token map is doomed to fail
 */
function is_token_bad(token){
    if (token === undefined){
        return 'undefined token';
    }
    var data = $page.token_data[token];
    if (data === undefined){
        return "no token data";
    }
    if (data.count == 0){
        return "no token points";
    }
    return false;
}

/** get_density_k adjusts the density map constant by the zoom state */
function get_density_k(){
    if ($const.DENSITY_MAP_ZOOM_DETAIL){
        return $const.ARRAY_FUZZ_DENSITY_CONSTANT * (1 << $state.zoom);
    }
    else {
        return $const.ARRAY_FUZZ_DENSITY_CONSTANT;
    }
}

/**zoomed_fuzz_array creates a 2d density map array
 *
 * @param x  coordinate of centre (points)
 * @param y  coordinate of centre
 * @param w  width of map (pixels)
 * @param h  height of map
 * @param points the points to map
 * @param zoom
 * @param k Gaussian fuzz constant
 * @param threshold value at outer edge of fuzz
 *
 * @return an unnormalised 2d floating point array representing the map
 */

function zoomed_fuzz_array(x, y, w, h, points, zoom, k, threshold){
    var scale = 1 << zoom;
    k /= (scale * scale);
    var r = calc_fuzz_radius(k, threshold);
    var z = get_zoom_point_bounds(zoom, x, y, w, h);
    var x_padding = r / z.x_scale;
    var y_padding = r / z.y_scale;
    points = bound_points(points, z.min_x - x_padding,
                          z.max_x + x_padding,
                          z.min_y - y_padding,
                          z.max_y + y_padding);
    var map = make_fuzz_array(points, k, threshold, w, h,
                              z.min_x, z.min_y,
                              z.x_scale, z.y_scale);
    return map;
}

/** extract_density_maps creates the density overlays in array form
 *
 * @param tokens list tokens to draw the overlays for
 * @param w width of density picture
 * @param h height of density picture
 * @param state a $state-alike object, saying where to zoom.
 *
 * @return a list of 2d array maps, or undefined on error.
 */

function extract_density_maps(tokens, w, h, state){
    log(tokens);
    var threshold = $const.ARRAY_FUZZ_DENSITY_THRESHOLD;
    var fatal_error = false;
    var html = "";
    var maps = [];
    var n = tokens.length;
    var k = get_density_k();
    for (var i = 0; i < n; i ++){
        var err = is_token_bad(tokens[i]);
        if (err){
            $(named_canvas("density_overlay")).css("visibility", "hidden");
            log(err, tokens[i]);
            html += "There is no data for <b>" + tokens[i] + "</b>.<br/>";
            fatal_error = true;
        }
        else {
            var data = $page.token_data[tokens[i]];
            var points = data.points;
            html += (data.count + " mentions of <br><b>'" + tokens[i] + "'</b>.<br/>");
            maps[i] = zoomed_fuzz_array(state.x, state.y, w, h, points,
                                        state.zoom, k, threshold);
        }
    }
    $("#token-notes").html(html);
    if (! fatal_error){
        return maps;
    }
    return undefined;
}

/** paint_density_duo performs an operation on two token maps and shows the result
 *
 * @param tokens list of tokens to use
 * @param op the operation function
 */
function paint_density_duo(tokens, op){
    var canvas = named_canvas("density_map", true, 0.25);
    var ctx = canvas.getContext("2d");
    var maps = extract_density_maps(tokens, canvas.width, canvas.height, $state);
    try {
        var m0 = maps[0];
        var m1 = maps[1];
        op(m0, m1, canvas.width, canvas.height);
        paste_density(ctx, m0);
    }
    catch (e){
        canvas.visibility = 'hidden';
        log(e);
    }
}

/** paint_density_top_n paints a density map based on the top few points
 *
 * @param token a token to map
 * @param n the number of points to use
 */

function paint_density_top_n(token, n){
    var canvas = named_canvas("density_map", true, 0.25);
    var ctx = canvas.getContext("2d");
    var threshold = $const.ARRAY_FUZZ_DENSITY_THRESHOLD;
    var k = get_density_k();
    var data = $page.token_data[token];
    var points = find_top_n_points(data.points, n);
    var html = ("using top " + n + " out of " + data.count +
                "mentions <br /> of <b>" + token + "</b>.<br/>");
    var map = zoomed_fuzz_array($state.x, $state.y, canvas.width, canvas.height,
                                points, $state.zoom, k, threshold);
    $("#token-notes").html(html);
    paste_density(ctx, map);
}

/** paint_density_uno performs an operation on a token map and shows the result
 *
 * the operation can be undefined, in which case the map is shown in original form
 *
 * @param token a token to map
 * @param op the operation function
 */
function paint_density_uno(token, op){
    var canvas = named_canvas("density_map", true, 0.25);
    var ctx = canvas.getContext("2d");
    var maps = extract_density_maps([token], canvas.width, canvas.height, $state);
    if (maps === undefined){
        return;
    }
    var map = maps[0];
    if (op !== undefined){
        try {
            op(map, canvas.width, canvas.height);
        }
        catch (e){
            canvas.visibility = 'hidden';
            log(e);
            return;
        }
    }
    paste_density(ctx, map);
}




/** paste_density turns a density array map into a canvas overlay
 */
function paste_density(ctx, map){
    paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);
    var canvas = apply_density_map(ctx);
    overlay(canvas);
}

/* density_uno_* operate on one map
 * density_duo_* operate on two
 */

/**density_uno_log transmutes the map into its log */
function density_uno_log(m0, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.log(r0[x]);
        }
    }
}

/**density_uno_exp uses the map values as exponent
 * The radix is low so that 32 bit floats don't saturate.
 */
function density_uno_exp(m0, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.pow(1.01, r0[x]);
        }
    }
}

/**density_uno_gate clips everything below a threshold it finds.
 */
function density_uno_gate(m0, width, height){
    /* first find a reasonable threshold */
    var sum = 0.0;
    var sqsum = 0.0;
    var count = 0;
    var x, y;
    for (y = 0; y < height; y++){
        var r0 = m0[y];
        for (x = 0; x < width; x++){
            var v = r0[x];
            if (v){
                sum += v;
                sqsum += v * v;
                count ++;
            }
        }
    }
    //var mean = sum / (width * height);
    var mean = sum / count;
    var variance = (sqsum - sum * mean) / count;

    /*hmm, now what?
     * cut out all below some point, but which.
     * the mean is a good start.*/
    var floor = mean;
    for (y = 0; y < height; y++){
        var r0 = m0[y];
        for (x = 0; x < width; x++){
            var v = Math.max(0, r0[x] - floor);
            r0[x] = v;
        }
    }
}

/** density_uno_cube cubes the map values */
function density_uno_cube(m0, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.pow(r0[x], 3);
        }
    }
}

/**density_uno_sqrt replaces map values with their positive square root*/
function density_uno_sqrt(m0, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.sqrt(r0[x]);
        }
    }
}


/**density_duo_mul multiplies two maps together */
function density_duo_mul(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] *= r1[x];
        }
    }
}

/**density_duo_add adds two maps together (unnormalised)*/
function density_duo_add(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] += r1[x];
        }
    }
}

/**density_duo_sub subtract the second map from the first */
function density_duo_sub(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.max(0, r0[x] - r1[x]);
        }
    }
}

/**density_duo_diff calculates the absolute difference between maps */
function density_duo_diff(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.abs(r0[x] - r1[x]);
        }
    }
}


function zoomed_paint(ctx, points, zoom, k, threshold, scale_args, max_height){
    $timestamp("start zoomed paint");
    var w = ctx.canvas.width;
    var h = ctx.canvas.height;
    var map = zoomed_fuzz_array($state.x, $state.y, w, h, points, zoom, k, threshold);
    max_height = paste_fuzz_array(ctx, map, scale_args, max_height);
    $timestamp("end zoomed paint");
    return max_height;
}

/** find_top_n_points
 *
 * @param points is a list of points
 * @param n is the number of top points you want
 * @return a list of the n highest valued points
 */
function find_top_n_points(points, n){
    var len = points.length;
    if (len == 0){
        return undefined;
    }
    var points2 = points.slice();
    points2.sort(function(a, b){
                     return a[2] - b[2];
                 });
    return points2.slice(-n);
}
