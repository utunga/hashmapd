/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

$const.DEBUG = true;

var $labels = {
    JSON_URL: "tokens/all-tokens-8-part-011.json",
    JSON_URL_TEMPLATE: "tokens/all-tokens-8-part-$$$.json",
    JSON_URL_COUNT_START: 1,
    JSON_URL_COUNT_STOP: 48,
    //JSON_URL_COUNT: 2,
    BITS: 7,
    COUNT_THRESHOLD: 150,
    FUZZ_DENSITY_CONSTANT: -0.012,
    FUZZ_DENSITY_THRESHOLD: 0.001,
    SIGNIFICANCE_THRESHOLD: 20,
    MIN_HEIGHT: 1.0,
    MAX_LABELS_PER_TOKEN: 5,
    DESIRED_JSON_ROWS: 10000,
    CLOSENESS_THRESHOLD: 10,

    token_stack: [],
    json_rows: [],
    peaks: [],
    descend_dont_climb: false,

    WIDTH: 128,
    HEIGHT: 128,
    x_scale: undefined,
    y_scale: undefined,

    draw_map: false,
    draw_map_for: '',
    stop_after_one: false,

    tokens_known: undefined
};

var find_label_peaks;

function label_gen(){
    /*load matching query parameters into $const, just this once. */
    interpret_query($const);
    interpret_query($labels);

    find_label_peaks =  $labels.descend_dont_climb ? find_label_peaks_descend : find_label_peaks_climb;

    $timestamp = get_timer();
    $waiters.tokens_known = $.Deferred();
    /* start downloading the main map */
    $waiters.map_known = get_json('locations', $const.MAP_RESOLUTION, hm_on_data);
    $.when($waiters.map_known).done(calc_label_scale, get_all_label_json);
    $page.loading = loading_screen();
    $.when($waiters.tokens_known).done(calculate_labels);

}


function calc_label_scale(){
    $labels.x_scale = $page.x_scale * $labels.WIDTH / $const.width;
    $labels.y_scale = $page.y_scale * $labels.HEIGHT / $const.height;
}

function get_all_label_json(){
    /* all requests at once, for parallelism?
     * or one after another, with pauses, for fewer "script is stuck" dialogs?
     *
     * ... try the latter first.
     */
    get_next_label_json($labels.JSON_URL_COUNT_START);
}

function get_next_label_json(n){
    var s = ("000" + n);
    s = s.substr(s.length - 3);
    var url = $labels.JSON_URL_TEMPLATE.replace('$$$', s);
    $timestamp("getting " + url);
    var d = $.getJSON(url, on_label_json);

    var cb;
    if (n < $labels.JSON_URL_COUNT_STOP){
        cb = function(){
            get_next_label_json(n + 1);
        };
    }
    else {
        cb = json_done;
    }
    d.done(cb);
}

function json_done(){
    $waiters.tokens_known.resolve();
}


function get_label_json(){
    var d = $.getJSON($labels.JSON_URL, on_label_json);
    d.done(
        function(){
            $waiters.tokens_known.resolve();
        }
    );
}

function maybe_store_token(token, count, points){
    if ((count < $labels.COUNT_THRESHOLD)
        || (token.substr(0,1) == '@')
        || (token.substr(0,4) == 'http')
        || (token.length > 15))
        return;
    $labels.token_stack.push([token, points, count]);
}


function on_label_json(data){
    var all_points = decode_points(data.rows);
    //XXX may need to sort the points.
    var i, p;
    //point: [x_coord, y_coord, value, precision, extra]
    var token = '';
    var count = 0, points = [];
    for (i = 0; i < all_points.length; i++){
        p = all_points[i];
        var t = p.pop()[0];
        if (t != token){
            maybe_store_token(token, count, points);
            count = 0;
            points = [];
            token = t;
        }
        points.push(p);
        count += p[2];
    }
    maybe_store_token(token, count, points);
}

function label_pixel_to_qt(x, y){
    x /= $labels.x_scale;
    y /= $labels.y_scale;
    x += $page.min_x;
    y += $page.min_y;
    return encode_point(x, y);
}


function calculate_labels(){
    $timestamp("calculating" + $labels.token_stack.length + "tokens");
    window.setTimeout(calc_one_label, 1);
}

function calc_one_label(){
    var d = $labels.token_stack.pop();
    var token = d[0];
    var points = d[1];
    var count = d[2];

    log(token, count);
    if($labels.draw_map_for != ''){
        $labels.draw_map = ($labels.draw_map_for == token);
    }
    var peaks = find_label_peaks(points, count);
    var n = Math.min(peaks.length, $labels.MAX_LABELS_PER_TOKEN);
    var labels = $labels.peaks;
    var found = 0;
    for (var i = 0; i < n; i++){
        var peak = peaks[i];
        //log(peak, peak.significance);
        if (peak.significance >= $labels.SIGNIFICANCE_THRESHOLD){
            /* two thirds found peak, 1 third centre of gravity */
            labels.push({x: (peak.x * 2 + peak.centre_x + 0.5) / 3,
                         y: (peak.y * 2 + peak.centre_y + 0.5) / 3,
                         size:  peak.size,
                         significance: peak.significance,
                         token: token
                        });
            found ++;
        }
    }
    //log("wanted", n, "found", found);

    if ($labels.token_stack.length == 0
        || $labels.stop_after_one){
        window.setTimeout(finish_calc_labels, 1);
    }
    else{
        window.setTimeout(calc_one_label, 1);
    }
}

function finish_calc_labels(){
    var peaks = winnow_peaks($labels.peaks);
    var rows = [];
    for (var i = 0; i < peaks.length; i++){
        var p = peaks[i];
        var coords = label_pixel_to_qt(p.x, p.y);
        coords.unshift(p.token);
        rows.push({key: coords,
                   value: p.size
                  });
    }
    $labels.json_rows = rows;

    $timestamp("finished calculating tokens");
    $("#content").append('<a id="label-json">download json</a>');
    $("#label-json").attr('href', 'data:'
                          //+ ','
                          + 'application/json;charset=utf-8,'
                          + JSON.stringify({rows: $labels.json_rows}));
}

function get_rows_map(peaks, name){
    var points = [];
    for (var i = 0, len = peaks.length; i < len; i++){
        var p = peaks[i];
        points.push([p.x / $labels.x_scale + $page.min_x,
                     p.y / $labels.y_scale + $page.min_y,
                     p.significance
                     //1
                    ]);
    }
    var map = make_label_fuzz_map(points);
    var canvas = named_canvas(name, false, $labels.WIDTH / $const.width);
    var ctx = canvas.getContext("2d");
    paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);
    return map;
}

function winnow_peaks(peaks){
    $timestamp("winnowing peaks");
    var i;
    var target = $labels.DESIRED_JSON_ROWS;
    log("want", target, "got", peaks.length);
    var map = get_rows_map(peaks, "all_labels");
    var max = 0.0;
    for (var y = 0; y < $labels.HEIGHT; y++){
        for (var x = 0; x < $labels.WIDTH; x++){
            if(map[y][x] > max)
                max = map[y][x];
        }
    }
    for (i = 0; i < peaks.length; i++){
        var p = peaks[i];
        //log(p.significance, map[parseInt(p.y)][parseInt(p.x)]);
        p.significance /= (max + map[parseInt(p.y)][parseInt(p.x)] * 3);
    }
    peaks.sort(function(a, b){return a.significance - b.significance});
    /*simple reduction: take potshots */
    while(peaks.length > target){
        var ii = Math.random() * Math.random();
        i = parseInt(ii * ii * peaks.length);
        peaks.splice(i, 1);
    };
    var map2 = get_rows_map(peaks, "filtered_labels");

    $timestamp("finished winnowing peaks");
    return peaks;
}


function find_peak(map){
    var pv = -1;
    var px = 0;
    var py = 0;
    /*XXX would be quicker to scan more sparsely and futz
     * around with an exact search at the end (given smooth landscape)*/
    for (var y = 0, ybound = $labels.HEIGHT; y < ybound; y++){
        for (var x = 0, xbound = $labels.WIDTH; x < xbound; x++){
            if (pv < map[y][x]){
                pv = map[y][x];
                py = y;
                px = x;
            }
        }
    }
    return {
        value: pv,
        x: px,
        y: py
    };
}

function descend_peak(map, peak, colour_pix){
    var current = [peak.value, peak.x, peak.y];
    map[peak.y][peak.x] = 0.0;
    var ymax = $labels.HEIGHT - 1;
    var xmax = $labels.WIDTH - 1;
    var sum = 0;
    var count = 0;
    var threshold = $labels.MIN_HEIGHT;

    //var gravity_x = 0;
    //var gravity_y = 0;
    var centre_x = 0.0;
    var centre_y = 0.0;

    //distances also?
    while (current.length){
        var next = [];
        count += current.length;
        for (var i = 0, len = current.length; i < len;){
            var v = current[i++];
            sum += v;
            var x = current[i++];
            var y = current[i++];
            centre_x += x * v;
            centre_y += y * v;

            if (y == 0 || y == ymax || x == 0 || x == xmax){
                //bugger the edge pixels! who cares!
                continue;
            }
            var v2 = map[y - 1][x];
            if (v2 < v && v2 > threshold){
                next.push(v2);
                next.push(x);
                next.push(y - 1);
                map[y - 1][x] = 0.0;
            }
            v2 = map[y + 1][x];
            if (v2 < v && v2 > threshold){
                next.push(v2);
                next.push(x);
                next.push(y + 1);
                map[y + 1][x] = 0.0;
            }
            v2 = map[y][x - 1];
            if (v2 < v && v2 > threshold){
                next.push(v2);
                next.push(x - 1);
                next.push(y);
                map[y][x - 1] = 0.0;
            }
            v2 = map[y][x + 1];
            if (v2 < v && v2 > threshold){
                next.push(v2);
                next.push(x + 1);
                next.push(y);
                map[y][x + 1] = 0.0;
            }
        }

        if (colour_pix !== undefined){
            for (i = 1, len = current.length; i < len; i += 3){
                var a = (current[i + 1] * $labels.WIDTH + current[i]) * 4;
                colour_pix[a] = peak.r;
                colour_pix[a + 1] = peak.g;
                colour_pix[a + 2] = peak.b;
                colour_pix[a + 3] = 255;
            }
        }
        current = next;
    }
    peak.centre_x = centre_x / sum;
    peak.centre_y = centre_y / sum;
    peak.sum = sum;
    peak.count = count;
}

function score_peak(peak, count){
    peak.significance = peak.sum / count;
    peak.size = parseInt(Math.pow(peak.significance * Math.log(peak.sum) * Math.sqrt(peak.value), 2));
}


function find_label_peaks_descend(points, count){
    //$timestamp("making label map");
    var labels = [];
    var i, j, x, y;
    var ymax = $labels.HEIGHT - 1;
    var xmax = $labels.WIDTH - 1;
    var map = make_label_fuzz_map(points);
    var pixels2;
    if ($labels.draw_map){
        $timestamp("made fuzz");
        var canvas = named_canvas("label_peaks", false, $labels.WIDTH / $const.width);
        var ctx = canvas.getContext("2d");
        paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);

        var canvas2 = named_canvas("colour_paths", false, $labels.WIDTH / $const.width);
        var ctx2 = canvas2.getContext("2d");
        var imgd2 = ctx2.getImageData(0, 0, $labels.WIDTH, $labels.HEIGHT);
        pixels2 = imgd2.data;
        $timestamp("paint fuzz");
    }

    var peaks = [];
    for (j = 0; j < 15; j++){
        var peak = find_peak(map);
        if (peak.value < 1)
            break;
        colour_peak(peak);
        descend_peak(map, peak, pixels2);
        score_peak(peak, count);
        //$timestamp("found peak " + j + " " + peak.x +',' + peak.y + ':' + peak.value);
        peaks.push(peak);
    }

    if ($labels.draw_map){
        ctx2.putImageData(imgd2, 0, 0);
        colour_peaks(ctx, peaks);
    }
    return blend_close_peaks(peaks);
}



function colour_peaks(ctx, peaks){
    var imgd = ctx.getImageData(0, 0, $labels.WIDTH, $labels.HEIGHT);
    var pixels = imgd.data;
    for (var i = 0; i < peaks.length; i++){
        var peak = peaks[i];
        var a = (parseInt(peak.y) * $labels.WIDTH + parseInt(peak.x)) * 4;
        //log(peak.x, peak.y, peak.centre_x, peak.centre_y);
        var b = (parseInt(peak.centre_y + 0.5) * $labels.WIDTH + parseInt(peak.centre_x + 0.5)) * 4;
        pixels[a] = 255;
        pixels[a + 1] = 127;
        pixels[a + 3] = 255;
        /*greeny for centre of gravity */
        pixels[b] = 125;
        pixels[b + 1] = 255;
        pixels[b + 3] = 255;
    }
    ctx.putImageData(imgd, 0, 0);
}

function colour_peak(peak){
    peak.r = parseInt(Math.random() * 255.9);
    peak.g = parseInt(Math.random() * 255.9);
    peak.b = parseInt(Math.random() * 255.9);
}

function make_label_fuzz_map(points){
    var map = make_fuzz_array(points,
                              $labels.FUZZ_DENSITY_CONSTANT,
                              $labels.FUZZ_DENSITY_THRESHOLD,
                              $labels.WIDTH, $labels.HEIGHT,
                              $page.min_x, $page.min_y,
                              $page.x_scale * $labels.WIDTH / $const.width,
                              $page.y_scale * $labels.HEIGHT / $const.height
                             );
    return map;
}


function find_label_peaks_climb(points, count){
    var i, j, x, y;
    var ymax = $labels.HEIGHT - 1;
    var xmax = $labels.WIDTH - 1;
    var map = make_label_fuzz_map(points);

    if ($labels.draw_map){
        var canvas = named_canvas("label_peaks", false, $labels.WIDTH / $const.width);
        var ctx = canvas.getContext("2d");
        paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);
    }

    var pathmap = [];
    for (y = 0; y < $labels.HEIGHT; y++){
        pathmap[y] = [];
    }

    var paths = [];
    var peaks = [];
    i = -1;
    var ix, iy;
    var min_height = $labels.MIN_HEIGHT;
    for (iy = 1; iy < ymax; iy += 1){
        for (ix = 1; ix < xmax; ix += 1){
            if (pathmap[iy][ix] === undefined && map[iy][ix] > min_height){
                i++;
                x = ix;
                y = iy;
                var p = {
                    //x: x,
                    //y: y,
                    weighted_x: 0.0,
                    weighted_y: 0.0,
                    sum: 0.0,
                    count: 1,
                    delegate: i
                    };
                paths[i] = p;
                while(1){
                    /*claim this point */
                    var current = map[y][x];
                    p.sum += current;
                    p.weighted_x += current * x;
                    p.weighted_y += current * y;
                    pathmap[y][x] = i;
                    /*now look for steepest uphill direction. In case of ties, the first
                     * checked direction wins, but ties are probably rare.*/
                    var dx, dy;
                    var best = current;
                    if (map[y - 1][x] > best){
                        dy = -1;
                        dx = 0;
                        best = map[y - 1][x];
                    }
                    if (map[y + 1][x] > best){
                        dy = 1;
                        dx = 0;
                        best = map[y + 1][x];
                    }
                    if (map[y][x - 1] > best){
                        dy = 0;
                        dx = -1;
                        best = map[y][x - 1];
                    }
                    if (map[y][x + 1] > best){
                        dy = 0;
                        dx = 1;
                        best = map[y][x + 1];
                    }
                    if (best == current){
                        /* this is a peak! */
                        p.x = x;
                        p.y = y;
                        p.value = current;
                        peaks.push(p);
                        break;
                    }
                    y += dy;
                    x += dx;

                    /*is this new point already part of a path? */
                    var path = pathmap[y][x];
                    if (path !== undefined){
                        var q = paths[path];
                        p.delegate = q.delegate;
                        p.x = q.x;
                        p.y = q.y;
                        var r = paths[q.delegate];
                        r.sum += p.sum;
                        r.weighted_x += p.weighted_x;
                        r.weighted_y += p.weighted_y;
                        r.count += p.count;
                        break;
                    }
                    /*abandon path if it reaches the edge. it just shouldn't do that. */
                    if (y == 0 || y == ymax || x == 0 || x == xmax){
                        log("edge point", x, y, "is uphill from somewhere");
                        break;
                    }
                }
            }
        }
    }


    var len = paths.length;
    for (i = 0; i < peaks.length; i++){
        var p = peaks[i];
        score_peak(p, count);
        //log(p.size, p.significance, count);
        p.centre_x = p.weighted_x / p.sum;
        p.centre_y = p.weighted_y / p.sum;
    }
    //peaks = blend_close_peaks(peaks);
    if ($labels.draw_map){
        var stride = $labels.WIDTH * 4;
        var canvas2 = named_canvas("colour_paths", false, $labels.WIDTH / $const.width);
        var ctx2 = canvas2.getContext("2d");
        var imgd2 = ctx2.getImageData(0, 0, $labels.WIDTH, $labels.HEIGHT);
        var pixels2 = imgd2.data;

        for (i = 0; i < peaks.length; i++){
            colour_peak(peaks[i]);
        }
        for (y = 0; y <= ymax; y++){
            for (x = 0; x <= xmax; x++){
                i = pathmap[y][x];
                if (i !== undefined){
                    var p = paths[i];
                    var q = paths[p.delegate];
                    var a = y * stride + x * 4;
                    pixels2[a] = q.r;
                    pixels2[a + 1] = q.g;
                    pixels2[a + 2] = q.b;
                    pixels2[a + 3] = 255;
                }
            }
        }
        ctx2.putImageData(imgd2, 0, 0);

        colour_peaks(ctx, peaks);
        colour_peaks(ctx2, peaks);
    }
    return blend_close_peaks(peaks);
}

function blend_close_peaks(peaks){
    var i, j, k;
    var out = [];
    var threshold = $labels.CLOSENESS_THRESHOLD * $labels.CLOSENESS_THRESHOLD;
    var pairs = [];
    var len = peaks.length;
    //log("initial scan, making groups");
    for (i = 0; i < len; i++){
        var p1 = peaks[i];
        var x = p1.x;
        var y = p1.y;
        //log(x, y, peaks[i].sum);
        for (j = i + 1; j < len; j++){
            var p2 = peaks[j];
            if ((x - p2.x) * (x - p2.x) + (y - p2.y) * (y - p2.y) < threshold){
                //log("combining", i, j);
                var group1 = p1.group;
                var group2 = p2.group;
                if (group1 === undefined && group2 == undefined){
                    p1.group = p2.group = {};
                    p1.group[i] = 1;
                    p1.group[j] = 1;
                }
                else if (group1 === group2){
                    //log("already joined!");
                    continue;
                }
                else if (group1 == undefined){
                    p1.group = group2;
                    group2[i] = 1;
                }
                else if (group2 == undefined){
                    p2.group = group1;
                    group1[j] = 1;
                }
                else {
                    for (m in group2){
                        group1[m] = 1;
                        peaks[m].group = group1;
                    }
                }
            }
        }
    }
    //log("merging groups");
    for (i = 0; i < len; i++){
        if (peaks[i] === undefined)
            continue;
        var group = peaks[i].group;
        if (group){
            //log("found a group", i);
            var sum = 0.0;
            var x = 0.0;
            var y = 0.0;
            var centre_x = 0.0;
            var centre_y = 0.0;
            for (m in group){
                var p = peaks[m];
                //log(m, p.x, p.y);
                var s = p.sum;
                sum += s;
                x += p.x * s;
                y += p.y * s;
                centre_x += p.centre_x * s;
                centre_y += p.centre_y * s;
                if (m != i){
                    peaks[m] = undefined;
                }
            }
            var combined = peaks[i];
            combined.sum = sum;
            combined.x = x / sum;
            combined.y = y / sum;
            combined.centre_x = centre_x / sum;
            combined.centre_y = centre_y / sum;
            delete peaks[i].group;
        }
    }

    for (i = len - 1; i >= 0; i--){
        if (peaks[i] === undefined){
            peaks.splice(i, 1);
            //log("removing", i);
        }
    }
    return peaks;
}