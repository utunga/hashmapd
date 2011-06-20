/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/*
            '+': density_add,
            '*': density_mul,
            '-': density_sub,
            '^': density_diff
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

function get_density_k(){
    if ($const.DENSITY_MAP_ZOOM_DETAIL){
        return $const.ARRAY_FUZZ_DENSITY_CONSTANT * (1 << $state.zoom);
    }
    else {
        return $const.ARRAY_FUZZ_DENSITY_CONSTANT;
    }
}

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
            html += (data.count + " recorded uses of <b>" + tokens[i] + "</b>.<br/>");
            maps[i] = zoomed_fuzz_array(state.x, state.y, w, h, points,
                                        state.zoom, k, threshold);
        }
    }
    $("#token-notes").html(html);
    if (! fatal_error){
        return maps;
    }
}

function paint_token_density(tokens){
    log("density_diff", tokens);
    var canvas = named_canvas("density_map", true, 0.25);
    var ctx = canvas.getContext("2d");
    var maps = extract_density_maps(tokens, canvas.width, canvas.height, $state);
    paste_density(ctx, maps[0]);
}

function paint_density_duo(args){
    log(args);
    var op = args.pop();
    var canvas = named_canvas("density_map", true, 0.25);
    var ctx = canvas.getContext("2d");
    var maps = extract_density_maps(args, canvas.width, canvas.height, $state);
    var m0 = maps[0];
    var m1 = maps[1];
    op(m0, m1, canvas.width, canvas.height);
    paste_density(ctx, m0);
}

function paint_density_uno(args){
    log(args);
    var op = args.pop();
    var canvas = named_canvas("density_map", true, 0.25);
    var ctx = canvas.getContext("2d");
    var maps = extract_density_maps(args, canvas.width, canvas.height, $state);
    op(maps[0], canvas.width, canvas.height);
    paste_density(ctx, maps[0]);
}

function paste_density(ctx, map){
    paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);
    var canvas2 = apply_density_map(ctx);
    overlay(canvas2);
}

function density_log(m0, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.log(r0[x]);
        }
    }
}

function density_sqrt(m0, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.sqrt(r0[x]);
        }
    }
}

function density_mul(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] *= r1[x];
        }
    }
}

function density_add(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] += r1[x];
        }
    }
}

function density_sub(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.max(0, r0[x] - r1[x]);
        }
    }
}

function density_diff(m0, m1, width, height){
    for (var y = 0; y < height; y++){
        var r0 = m0[y];
        var r1 = m1[y];
        for (var x = 0; x < width; x++){
            r0[x] = Math.abs(r0[x] - r1[x]);
        }
    }
}


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

function zoomed_paint(ctx, points, zoom, k, threshold, scale_args, max_height){
    $timestamp("start zoomed paint");
    var w = ctx.canvas.width;
    var h = ctx.canvas.height;
    var map = zoomed_fuzz_array($state.x, $state.y, w, h, points, zoom, k, threshold);
    $timestamp("made zoomed map");
    max_height = paste_fuzz_array(ctx, map, scale_args, max_height);
    $timestamp("pasted zoomed map");
    return max_height;
}
