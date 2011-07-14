/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

$const.DEBUG = true;

var $labels = {
    JSON_URL: "tokens/all-tokens-8-part-011.json",
    JSON_URL_TEMPLATE: "tokens/all-tokens-8-part-$$$.json",
    JSON_URL_COUNT: 48,
    //JSON_URL_COUNT: 2,
    BITS: 7,
    THRESHOLD: 400,
    FUZZ_DENSITY_CONSTANT: -0.012,
    FUZZ_DENSITY_THRESHOLD: 0.001,
    VOTE_THRESHOLD: 200,

    token_stack: [],
    json_rows: [],
    descend_dont_climb: true,

    WIDTH: 128,
    HEIGHT: 128,
    x_scale: undefined,
    y_scale: undefined,

    draw_map: true,
    stop_after_one: true,

    tokens_known: undefined
};

function label_gen(){
    /*load matching query parameters into $const, just this once. */
    interpret_query($const);
    interpret_query($labels);

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
    get_next_label_json(1);
}

function get_next_label_json(n){
    var s = ("000" + n);
    s = s.substr(s.length - 3);
    var url = $labels.JSON_URL_TEMPLATE.replace('$$$', s);
    $timestamp("getting " + url);
    var d = $.getJSON(url, on_label_json);

    var cb;
    if (n < $labels.JSON_URL_COUNT){
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
    if ((count < $labels.THRESHOLD)
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
    //log(x, y);
    x /= $labels.x_scale;
    y /= $labels.y_scale;
    x += $page.min_x;
    y += $page.min_y;
    //log(x, y);
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
    var peaks;
    if ($labels.descend_dont_climb)
        peaks = find_label_peaks(points, count);
    else
        peaks = find_label_peaks2(points, count);
    var rows = $labels.json_rows;
    for (var i = 0; i < peaks.length; i++){
        var p = peaks[i];
        var coords = label_pixel_to_qt(p[0], p[1]);
        coords.unshift(token);
        rows.push({key: coords,
                   value: p[2]
                  });
    }

    if ($labels.token_stack.length == 0
        || $labels.stop_after_one){
        window.setTimeout(finish_calc_labels, 1);
    }
    else{
        window.setTimeout(calc_one_label, 1);
    }
}

function finish_calc_labels(){
    $timestamp("finished calculating tokens");
    $("#content").append('<a id="label-json">download json</a>');
    $("#label-json").attr('href', 'data:'
                          //+ ','
                          + 'application/json;charset=utf-8,'
                          + JSON.stringify({rows: $labels.json_rows}));
}


function find_peak(map){
    var pv = -1;
    var px = 0;
    var py = 0;
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
    var threshold = 0.05;
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
            //gravity_x += v / (x - peak.x);
            //gravity_y += v / (y - peak.y);
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
            //log("colouring", current.length, peak.r, peak.g, peak.b);
            for (i = 1, len = current.length; i < len; i += 3){
                var a = (current[i + 1] * $labels.WIDTH + current[i]) * 4;
                //log(a);
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

function score_peak(peak){
    peak.score = peak.sum * peak.count * Math.log(peak.value);
}


function find_label_peaks(points, count){
    //$timestamp("making label map");
    var labels = [];
    var i, j, x, y;
    var ymax = $labels.HEIGHT - 1;
    var xmax = $labels.WIDTH - 1;
    var map = make_fuzz_array(points,
                              $labels.FUZZ_DENSITY_CONSTANT,
                              $labels.FUZZ_DENSITY_THRESHOLD,
                              $labels.WIDTH, $labels.HEIGHT,
                              $page.min_x, $page.min_y,
                              $labels.x_scale,
                              $labels.y_scale
                             );
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
        score_peak(peak);
        //$timestamp("found peak " + j + " " + peak.x +',' + peak.y + ':' + peak.value);
        peaks.push(peak);
    }

    if ($labels.draw_map){
        ctx2.putImageData(imgd2, 0, 0);

        var imgd = ctx.getImageData(0, 0, $labels.WIDTH, $labels.HEIGHT);
        var pixels = imgd.data;
        for (i = 0; i < peaks.length; i++){
            var peak = peaks[i];
            var a = (peak.y * $labels.WIDTH + peak.x) * 4;
            log(peak.x, peak.y, peak.centre_x, peak.centre_y);
            var b = (parseInt(peak.centre_y) * $labels.WIDTH + parseInt(peak.centre_x)) * 4;
            pixels[a] = 255;
            pixels[a + 1] = 127;
            pixels[a + 3] = 255;
            pixels[b] = 125;
            pixels[b + 1] = 255;
            pixels[b + 3] = 255;
        }
        ctx.putImageData(imgd, 0, 0);
    }

    var n = Math.min(peaks.length, 5);
    for (i = 0; i < n; i++){
        var peak = peaks[i];
        y = (peak.y + peak.centre_y + 0.5) * 0.5;
        x = (peak.x + peak.centre_x + 0.5) * 0.5;
        //log(peak.x, peak.centre_x, x, peak.y, peak.centre_y, y);
        labels.push([x, y, peak.score]);
    }
    return labels;
}

function colour_peak(peak){
    peak.r = parseInt(Math.random() * 255.9);
    peak.g = parseInt(Math.random() * 255.9);
    peak.b = parseInt(Math.random() * 255.9);
}


function find_label_peaks2(points, count){
    var labels = [];
    var i, j, x, y;
    var ymax = $labels.HEIGHT - 1;
    var xmax = $labels.WIDTH - 1;

    var map = make_fuzz_array(points,
                              $labels.FUZZ_DENSITY_CONSTANT,
                              $labels.FUZZ_DENSITY_THRESHOLD,
                              $labels.WIDTH, $labels.HEIGHT,
                              $page.min_x, $page.min_y,
                              $page.x_scale * $labels.WIDTH / $const.width,
                              $page.y_scale * $labels.HEIGHT / $const.height
                             );

    if ($labels.draw_map){
        var canvas = named_canvas("label_peaks", false, $labels.WIDTH / $const.width);
        var ctx = canvas.getContext("2d");
        paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);
    }

    var locations = [];
    for (y = 1; y < ymax; y += 4){
        for (x = 1; x < xmax; x += 4){
            if (map[y][x] > 0.05){
                locations.push(
                    {
                        x: x,
                        y: y,
                        sum: map[y][x],
                        count: 1
                    });
            }
        }
    }
    shuffle(locations);

    var paths = [];
    for (y = 0; y < $labels.HEIGHT; y++){
        paths[y] = [];
    }

    var len = locations.length;

    for (i = 0; i < len; i++){
        var p = locations[i];
        x = p.x;
        y = p.y;
        p.delegate = i;
        p.id = i;
        while(1){
            if (y == 0 || y == ymax || x == 0 || x == xmax){
                /*abandon path when it reaches the edge. it just shouldn't. */
                log("edge point", x, y, "is uphill from somewhere");
                break;
            }
            var loc = paths[y][x];
            if (loc){
                var q = locations[loc];
                p.delegate = q.delegate;
                q.sum += p.sum;
                q.count += p.count;
                break;
            }
            paths[y][x] = i;
            var current = map[y][x];
            var dx = 0, dy = 0;
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
                break;
            }
            p.sum += best;
            y += dy;
            x += dx;
        }
    }

    var votes = {};
    var p;
    for (i = 0; i < len; i++){
        p = locations[i];
        var q = p.delegate;
        votes[q] = (votes[q] || 0.0) + p.sum;
    }
    //$timestamp("counted votes");
    var _keys = [];
    for (var k in votes) _keys.push(k);
    log(_keys);
    for (i in votes){
        p = locations[i];
        log(len, i, p);
        var score = votes[i] * Math.log(map[p.y][p.x]) * Math.log(count);
        if (score >= $labels.VOTE_THRESHOLD){
            labels.push([p.x, p.y, score]);
        }
    }

    if ($labels.draw_map){
        var w = $labels.WIDTH;
        var stride = w * 4;
        var imgd = ctx.getImageData(0, 0, w, $labels.HEIGHT);
        var pixels = imgd.data;


        var canvas2 = named_canvas("colour_paths", false, $labels.WIDTH / $const.width);
        var ctx2 = canvas2.getContext("2d");
        var imgd2 = ctx2.getImageData(0, 0, $labels.WIDTH, $labels.HEIGHT);
        var pixels2 = imgd2.data;

        for (i in votes){
            colour_peak(locations[i]);
        }
        for (y = 0; y <= ymax; y++){
            for (x = 0; x <= xmax; x++){
                i = paths[y][x];
                if (i !== undefined){
                    p = locations[i];
                    var q = locations[p.delegate];
                    var a = y * stride + x * 4;
                    pixels2[a] = q.r;
                    pixels2[a + 1] = q.g;
                    pixels2[a + 2] = q.b;
                    pixels2[a + 3] = 255;
                }
            }
        }
        ctx2.putImageData(imgd2, 0, 0);

        for (i in votes){
            p = locations[i];
            var score = votes[i] * Math.log(map[p.y][p.x]) * Math.log(count);
            log(parseInt(score), p.x, p.y, votes[i], map[p.y][p.x]);
            pixels[p.y * stride + p.x * 4] = 255;
            pixels[p.y * stride + p.x * 4 + 3] = 255;
        }
        ctx.putImageData(imgd, 0, 0);
    }
    return labels;
}

/* shuffle an array in place */
function shuffle(array) {
    var tmp, current;
    var top = array.length;
    if (top == 0)
        return;
    while(--top){
        current = Math.floor(Math.random() * (top + 1));
        tmp = array[current];
        array[current] = array[top];
        array[top] = tmp;
    }
}
