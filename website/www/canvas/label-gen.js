/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

$const.DEBUG = true;

var $labels = {
    JSON_URL: "tokens/all-tokens-8-part-011.json",
    JSON_URL_TEMPLATE: "tokens/all-tokens-8-part-$$$.json",
    JSON_URL_COUNT: 48,
    //JSON_URL_COUNT: 5,
    BITS: 7,
    THRESHOLD: 400,
    FUZZ_DENSITY_CONSTANT: -0.005,
    FUZZ_DENSITY_THRESHOLD: 0.001,
    VOTE_THRESHOLD: 200,

    token_stack: [],
    json_rows: [],

    WIDTH: 128,
    HEIGHT: 128,
    x_scale: undefined,
    y_scale: undefined,

    draw_map: false,

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
    var peaks = find_label_peaks(points, count);
    var rows = $labels.json_rows;
    for (var i = 0; i < peaks.length; i++){
        var p = peaks[i];
        var coords = label_pixel_to_qt(p[0], p[1]);
        coords.unshift(token);
        rows.push({key: coords,
                   value: p[2]
                  });
    }

    if ($labels.token_stack.length == 0){
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
                          + ','
                          //+ 'application/json;charset=utf-8,'
                          + JSON.stringify({rows: $labels.json_rows}));
}


function find_label_peaks(points, count){
    //$timestamp("making label map");
    var map = make_fuzz_array(points,
                              $labels.FUZZ_DENSITY_CONSTANT,
                              $labels.FUZZ_DENSITY_THRESHOLD,
                              $labels.WIDTH, $labels.HEIGHT,
                              $page.min_x, $page.min_y,
                              $page.x_scale * $labels.WIDTH / $const.width,
                              $page.y_scale * $labels.HEIGHT / $const.height
                             );
    //$timestamp("made label map");
    if ($labels.draw_map){
        var canvas = named_canvas("label_peaks", false, $labels.WIDTH / $const.width);
        var ctx = canvas.getContext("2d");
        paste_fuzz_array(ctx, map, $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS);
        //$timestamp("pasted fuzz");
    }
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
        p = locations[i];
        var score = votes[i] * Math.log(map[p[1]][p[0]]) * Math.log(count);
        if (score >= $labels.VOTE_THRESHOLD){
            labels.push([p[0], p[1], score]);
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
