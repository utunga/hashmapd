/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

$const.DEBUG = true;

var $labels = {
    JSON_URL: "tokens/all-tokens-8-part-011.json",
    BITS: 7,
    THRESHOLD: 100,
    FUZZ_DENSITY_CONSTANT: -0.005,
    FUZZ_DENSITY_THRESHOLD: 0.001,
    VOTE_THRESHOLD: 100,

    WIDTH: 128,
    HEIGHT: 128,
    x_scale: undefined,
    y_scale: undefined,


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
    $.when($waiters.map_known).done(get_label_json, calc_label_scale);
    $page.loading = loading_screen();
    $.when($waiters.tokens_known).done(calculate_labels);
}

function calc_label_scale(){
    $labels.x_scale = $page.x_scale * $labels.WIDTH / $const.width;
    $labels.y_scale = $page.y_scale * $labels.HEIGHT / $const.height;
}


function get_label_json(){
    var d = $.getJSON($labels.JSON_URL, on_label_json);
    d.done(
        function(){
            $waiters.tokens_known.resolve();
        }
    );
}

function on_label_json(data){
    log("in on_label_json");
    var points = decode_points(data.rows);
    //XXX may need to sort the points.
    var i, p;
    var cache = $page.token_data;
    //point: [x_coord, y_coord, value, precision, extra]
    var token = '';
    var count, points2;
    for (i = 0; i < points.length; i++){
        p = points[i];
        var t = p.pop()[0];
        if (t != token){
            //log(token, count);
            if (count >= $labels.THRESHOLD){
                cache[token] = {
                    count: count,
                    points: points2
                };
            }
            count = 0;
            points2 = [];
            token = t;
        }
        points2.push(p);
        count += p[2];
    }
    if (count >= $labels.THRESHOLD){
        cache[token] = {
            count: count,
            points: points2
        };
    }
}

function label_pixel_to_qt(x, y){
    log(x, y);
    x /= $labels.x_scale;
    y /= $labels.y_scale;
    x += $page.min_x;
    y += $page.min_y;
    log(x, y);
    return encode_point(x, y);
}


function calculate_labels(){
    log("in calculate_labels");
    var i, p;
    var cache = $page.token_data;
    var token;
    var rows = [];
    for (token in cache){
        var peaks = find_label_peaks(cache[token]);
        for (i = 0; i < peaks.length; i++){
            p = peaks[i];
            var coords = label_pixel_to_qt(p[0], p[1]);
            coords.unshift(token);
            rows.push({key: coords,
                       value: p[2]
                      });
        }
        //break;
    }
    $("#content").append('<a id="label-json">download json</a>');
    $("#label-json").attr('href', 'data:'
                          + ','
                          //+ 'application/json;charset=utf-8,'
                          + JSON.stringify({rows: rows}));
}

function find_label_peaks(data){
    //var points = data.points;
    //var count = data.count;
    //$timestamp("making label map");
    var map = make_fuzz_array(data.points,
                              $labels.FUZZ_DENSITY_CONSTANT,
                              $labels.FUZZ_DENSITY_THRESHOLD,
                              $labels.WIDTH, $labels.HEIGHT,
                              $page.min_x, $page.min_y,
                              $page.x_scale * $labels.WIDTH / $const.width,
                              $page.y_scale * $labels.HEIGHT / $const.height
                             );
    //$timestamp("made label map");

    var canvas = named_canvas("label_peaks", false, $labels.WIDTH / $const.width);
    var ctx = canvas.getContext("2d");
    paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);
    //$timestamp("pasted fuzz");

    var locations = [];
    var i, x, y;
    var ymax = $labels.HEIGHT - 1;
    var xmax = $labels.WIDTH - 1;
    for (y = 1; y < ymax; y++){
        for (x = 1; x < xmax; x++){
            if (map[y][x] != 0.0){
                locations.push([x, y]);
            }
        }
    }
    //$timestamp("made locations");

    shuffle(locations);
    //$timestamp("shuffled locations");


    var paths = [];
    for (y = 0; y < $labels.HEIGHT; y++){
        paths[y] = [];
    }
    //$timestamp("made paths");

    var len = locations.length;

    for (i = 0; i < len; i++){
        var p = locations[i];
        x = p[0];
        y = p[1];
        while(1){
            if (paths[y][x]){
                p[2] = locations[paths[y][x]][2];
                break;
            }
            paths[y][x] = i;

            var current = map[y][x];
            var dx = 0, dy = 0;
            var best = current;
            var d = -1;
            /* these tedious bounds checks are *almost* unnecessary: the
             * peak of a term can't be right on the very edge, except in
             * bizzare circumstances, and in the x case, the comparison
             * against undefined would do the right thing anyway.
             */
            if (y > 0 && map[y - 1][x] > best){
                dy = -1;
                dx = 0;
                best = map[y - 1][x];
            }
            if (y < ymax && map[y + 1][x] > best){
                dy = 1;
                dx = 0;
                best = map[y + 1][x];
            }
            if (x > 0 && map[y][x - 1] > best){
                dy = 0;
                dx = -1;
                best = map[y][x - 1];
            }
            if (x < xmax && map[y][x + 1] > best){
                dy = 0;
                dx = 1;
                best = map[y][x + 1];
            }
            if (dx == 0 && dy == 0){
                /* this is a peak! */
                p[0] = x;
                p[1] = y;
                p[2] = i;
                break;
            }
            y += dy;
            x += dx;
        }
    }
    //$timestamp("climbed hills");


    var votes = {};
    var p;
    for (i = 0; i < len; i++){
        p = locations[i];
        var delegate = p[2];


        if (votes[delegate] == undefined){
            votes[delegate] = 1;
            continue;
        }
        votes[delegate]++;
    }
    //$timestamp("counted votes");
    var labels = [];

    for (i in votes){
        if (votes[i] >= $labels.VOTE_THRESHOLD){
            p = locations[i];
            labels.push([p[0], p[1], votes[i]]);
        }
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
