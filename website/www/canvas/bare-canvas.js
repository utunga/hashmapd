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
    DATA_URL: 'locations-9.json',
    TOKEN_DENSITY_URL: 'token_density-8.json',
    LABELS_URL: 'tokens-7.json',
    //DATA_URL: 'http://hashmapd.couchone.com/frontend_dev/_design/user/_view/xy_coords?group=true',
    PADDING: 16,    /*padding for the image as a whole. */
    FUZZ_CONSTANT: -0.37, /*concentration of peaks, negative inverse variance */
    FUZZ_OFFSET: 0.5, /* lift floor by this much (0.5 rounds, more to lengthen tails) */
    FUZZ_PER_POINT: 8, /* a single point generates this much fuzz */
    FUZZ_MAX_RADIUS: 16, /*fuzz never reaches beyond this far */
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



    trailing_commas_are_GOOD: true
};

/** hm_draw_map is the main entrance point.
 *
 * Nothing much happens until the json is loaded.
 */

function hm_draw_map(){
    $hm.timer = {start: Date.now()};
    $hm.canvas = fullsize_canvas();
    $hm.map_known = $.Deferred();
    $hm.map_drawn = $.Deferred();
    $hm.have_labels = $.Deferred();
    $hm.have_density = $.Deferred();
    start_fuzz_creation();

    $.getJSON($hm.DATA_URL, function(data){
                  hm_on_data(data);
              });

    $.getJSON($hm.TOKEN_DENSITY_URL, function(data){
                  hm_on_token_density(data);
              });

    if (get_query()["labels"]){
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
    $hm.timer.pre_fuzz = Date.now();
    var fuzz = make_fuzz($hm.FUZZ_MAX_MULTIPLE,
                         $hm.FUZZ_MAX_RADIUS,
                         $hm.FUZZ_CONSTANT,
                         $hm.FUZZ_OFFSET,
                         $hm.FUZZ_PER_POINT);
    $hm.timer.post_fuzz = Date.now();
    $hm.hill_fuzz = fuzz;
}

function paint_map(){
    $hm.hill_fuzz.ready.then(_paint_map);
}

function paint_labels(){
    $hm.map_drawn.then(_paint_labels);
}

function paint_density_map(){
    $hm.map_drawn.then(
        function(){
            $hm.hill_fuzz.ready.then(_paint_density_map);
        }
    );
}


/** new_canvas makes a canvas of the requested size.
 *
 */
function new_canvas(width, height, id){
    var canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    if (id){
        canvas.id = id;
    }
    return canvas;
}

/** fullsize_canvas helpfully makes a canvas as big as the main map
 *
 * In development it pastes the canvas onto the webpage/
 */
function fullsize_canvas(){
    var canvas = new_canvas($hm.width + 2 * $hm.PADDING,
                            $hm.height + 2 * $hm.PADDING);
    document.getElementById("content").appendChild(canvas);
    return canvas;
}


/*XXX ignoring cases where CSS pixels are not device pixels */

/** Make_fuzz makes little fuzzy circle images for point convolution
 *
 * The returned object contains references to html imgs index by
 * numbers from 1 to <densities>, which contain fuzz corresponding to
 * that number of points.  The images vary in size, being clipped at
 * the edge of the representable curve -- higher densities spread
 * slightly further.
 *
 * The images are not necessarily ready when the function returns.
 * You can use any individually via its .onload() handler or after
 * checking its .complete attribute, but the simplest thing is to use
 * the returned object's .ready jquery deferred attribute.  Like so:
 *
 * make_fuzz(...).ready.then(whatever);
 *
 * The fuzz is approximately gaussian and carried in the images alpha
 * channel.
 *
 * Look at the attributes of the $hm object that start with FUZZ_ for
 * documentation and useful values.
 *
 * The formula used is Math.exp(k * d) + <floor>
 *
 * @param densities make fuzz images with desin=ties from 1 to <densities>
 * @param radius    size of guassian table. no image will exceed this.
 * @param k         negative inverse variance. closer to 0 is flatter.
 * @param offset    lifts floor in truncating (0.5 rounds, more to lengthen tails)
 * @param intensity level of fuzz at the center of density 1.
 *
 * @return an object referencing images that *will* contain fuzz when it it is ready.
 */

function make_fuzz(densities, radius, k, offset, intensity){
    var x, y;
    /* work out a table 0-1 fuzz values */
    var table = [];
    /* middle pixel + radius on either side */
    var tsize = 2 * radius + 1;
    for (y = 0; y < tsize; y++){
        table[y] = [];
        var ty = table[y];
        var dy2 = (y - radius) * (y - radius);
        for (x = 0; x < tsize; x++){
            var dx2 = (x - radius) * (x - radius);
            ty[x] = Math.exp(Math.sqrt(dx2 + dy2) * k);
        }
    }
    var centre_row = table[radius];
    var images = {loaded: 0,
                  ready: $.Deferred()
                 };

    for (var i = 1; i <= densities; i++){
        var peak = intensity * i;
        /* find how wide it needs to be. We want to clip <clip> off
         * each side of the table.*/
        for (var clip = 0; clip < radius; clip++){
            var value = centre_row[clip] * peak + offset;
            if (value >= 1){
                break;
            }
        }
        var size = tsize - 2 * clip;
        var canvas = new_canvas(size, size);
        var ctx = canvas.getContext("2d");
        var imgd = ctx.getImageData(0, 0, size, size);
        var pixels = imgd.data;
        var stride = size * 4;
        for (y = 0; y < size; y++){
            var sy = clip + y;
            var row = y * stride;
            for (x = 0; x < size; x++){
                var sx = clip + x;
                var a = parseInt(table[sy][sx] * peak + offset);
                var p = row + x * 4;
                pixels[p] = 255;
                pixels[p + 3] = a;
            }
        }
        ctx.putImageData(imgd, 0, 0);
        var img = document.createElement("img");
        img.id = "fuzz-" + i + "-radius-" + (size - 1) / 2;
        images[i] = img;
        /*hacky way to forward onload */
        img.onload = function(){
            images.loaded++;
            if(images.loaded == densities){
                images.ready.resolve();
            }
        };
        img.src = canvas.toDataURL();
        $("#helpers").append(img);
    }
    return images;
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


function decode_and_filter_points(raw, xmin, xmax, ymin, ymax){
    var i, j;
    var points = [];
    if ($hm.USING_QUAD_TREE){
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
    }
    else {
        for (i = 0; i < raw.length; i++){
            var r = raw[i];
            points.push([r.key[0], r.key[1],  r.value]);
        }
    }

    /*passing straight through is a common case*/
    if (xmin === undefined &&
        xmax === undefined &&
        ymin === undefined &&
        ymax === undefined){
        return points;
    }
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
    $hm.timer.on_data = Date.now();
    var i;
    var width = $hm.width;
    var height = $hm.width;
    var max_value = 0;
    var max_x = -1e999;
    var max_y = -1e999;
    var min_x =  1e999;
    var min_y =  1e999;
    var points = decode_and_filter_points(data.rows);
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
    $hm.x_scale = width / $hm.range_x;
    $hm.y_scale = height / $hm.range_y;
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
    $hm.timer.fuzz_ready = Date.now();
    paste_fuzz(fuzz_ctx, points, $hm.hill_fuzz);
    $hm.timer.fuzz_pasted = Date.now();
    hillshading(fuzz_ctx, ctx, 1, Math.PI * 1 / 4, Math.PI / 4);
    $hm.timer.hillshaded = Date.now();
    $hm.map_drawn.resolve();
}

function paste_fuzz(ctx, points, images){
    var counts = [];
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        var x = $hm.PADDING + (p[0] - $hm.min_x) * $hm.x_scale;
        var y = $hm.PADDING + (p[1] - $hm.min_y) * $hm.y_scale;
        var count = p[2];
        counts[count] = (counts[count] || 0) + 1;
        var img;
        if (count <= $hm.FUZZ_MAX_MULTIPLE){
            img = images[count];
        }
        else{
            /* XXX jump up to next scale */
            img = images[$hm.FUZZ_MAX_MULTIPLE];
        }
        ctx.drawImage(img, x - img.width / 2, y - img.height / 2);
    }
}

function hillshading(map_ctx, target_ctx, scale, angle, alt){
    var canvas = target_ctx.canvas;
    var width = canvas.width;
    var height = canvas.height;
    var map_imgd = map_ctx.getImageData(0, 0, width, height);
    var map_pixels = map_imgd.data;

    /*colour in the sea in one go */
    target_ctx.fillStyle = "rgb(147,187,189)";
    target_ctx.fillRect(0, 0, width, height);
    var target_imgd = target_ctx.getImageData(0, 0, width, height);
    var target_pixels = target_imgd.data;

    scale = 1.0 / (8.0 * scale);
    var sin_alt = Math.sin(alt);
    var cos_alt = Math.cos(alt);
    var perpendicular = angle - Math.PI / 2;
    var stride = width * 4;
    var colours = make_colour_range_mountains(115);
    var row = stride; /*start on row 1, not row 0 */
    for (var y = 1, yend = height - 1; y < yend; y++){
        for (var x = 4 + 3, xend = stride - 4; x < xend; x += 4){
            var a = row + x;
            var cc = map_pixels[a];
            if (cc < 1){
                continue;
            }

            var tc = map_pixels[a - stride];
            var tl = map_pixels[a - stride - 4];
            var tr = map_pixels[a - stride + 4];
            var cl = map_pixels[a - 4];
            var cr = map_pixels[a + 4];
            var bc = map_pixels[a + stride];
            var bl = map_pixels[a + stride - 4];
            var br = map_pixels[a + stride + 4];

            /* Slope -- there may be a quicker way of calculating sin/cos */
            var dx = ((tl + 2 * cl + bl) - (tr + 2 * cr + br)) * scale;
            var dy = ((bl + 2 * bc + br) - (tl + 2 * tc + tr)) * scale;
            var slope = Math.PI / 2 - Math.atan(Math.sqrt(dx * dx + dy * dy));

            var sin_slope = Math.sin(slope);
            var cos_slope = Math.cos(slope);

            /* Aspect */
            var aspect = Math.atan2(dx, dy);

            /* Shade value */
            var c = Math.max(sin_alt * sin_slope + cos_alt * cos_slope *
                             Math.cos(perpendicular - aspect), 0);
            var colour = colours[cc];
            if (cc == 1){ /* the sea shore has less shadow */
                c = (0.5 + c) / 2;
            }
            target_pixels[a - 3] = colour[0] + 140 * c;
            target_pixels[a - 2] = colour[1] + 140 * c;
            target_pixels[a - 1] = colour[2] + 130 * c;
            target_pixels[a] = 255;
        }
        row += stride;
    }
    target_ctx.putImageData(target_imgd, 0, 0);
}


/** make_colour_range_mountains utility
 *
 * @param scale range for rgb values (e.g. 255 for full range)
 * @return a 256 long array of colours.
 *
 * near sea - yellow, brown
 * then bright green
 * then darker green
 * then brown
 * then grey
 * ?
 */
function make_colour_range_mountains(scale){
    var colours = [];
    var checkpoints = [
      /* r, g, b are out of 100 */
      /*   r    g    b  height  */
        [ 80,  90,  90,   0],
        [ 70,  90,  90,   1],
        [ 95,  90,  20,   2],
        [ 60, 105,  10,  10],
        [ 40,  60,  10,  60],
        [ 45,  40,   0, 100],
        [ 45,  45,  35, 150],
        [ 95,  95,  95, 160],
        [100, 100, 100, 255]
    ];
    var i = 0;
    for (var j = 0; j < checkpoints.length - 1; j++){
        var src = checkpoints[j];
        var dest = checkpoints[j + 1];
        var r = src[0];
        var g = src[1];
        var b = src[2];
        var start = src[3];
        /*destinations*/
        var r2 = dest[0];
        var g2 = dest[1];
        var b2 = dest[2];
        var end = dest[3];
        /*deltas*/
        var steps = end - start;
        var dr = (r2 - r) / steps;
        var dg = (g2 - g) / steps;
        var db = (b2 - b) / steps;
        for (i = start; i < end; i++){
            colours.push([parseInt(r / 100 * scale),
                          parseInt(g / 100 * scale),
                          parseInt(b / 100 * scale)]);
            r += dr;
            g += dg;
            b += db;
        }
    }
    colours.push([parseInt(r / 100 * scale),
                  parseInt(g / 100 * scale),
                  parseInt(b / 100 * scale)]);
    return colours;
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
    $hm.timer.doing_tokens = Date.now();
    var i;
    var points = decode_and_filter_points(data.rows,
                                          $hm.min_x, $hm.max_x,
                                          $hm.min_y, $hm.max_y);

    var token_canvas = fullsize_canvas();
    var token_ctx = token_canvas.getContext("2d");

    var max_freq = 0;
    for (var i = 0; i < points.length; i++){
        var freq = points[i][2];
        max_freq = Math.max(freq, max_freq);
    }
    $hm.hill_fuzz.ready.then(
        function(){
            paste_fuzz(token_ctx, points, $hm.hill_fuzz);
            hm_timer_results();
        }
    );
    $hm.have_density.resolve();
}

function _paint_density_map(){
}



/*don't do too much until the drawing is done.*/

function hm_on_labels(data){
    var points = decode_and_filter_points(data.rows,
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

function add_label(ctx, text, x, y, size, colour, shadow){
    if (colour === undefined){
        colour = "#000";
    }
    if (shadow && size > 8){
        ctx.shadowColor = shadow;
        ctx.shadowBlur = size * 0.25;
    }
    ctx.font = size + "px sans-serif";
    ctx.fillStyle = colour;
    ctx.fillText(text, x, y);
}


function hm_timer_results(){
    var k, v, ordered = [];
    for (k in $hm.timer){
        v = $hm.timer[k];
        ordered.push([v, k]);
    }
    ordered.sort();
    var s = "<table><tr><td><td>time<td>delta";
    var t2 = 0;
    for (var i = 0; i < ordered.length; i++){
        var t = ordered[i][0] - ordered[0][0];
        var d = t - t2;
        t2 = t;
        s += "<tr><td>" + ordered[i][1] + "<td>" + t + "<td>" + d + "\n";
    }
    s += "</table>";
    //alert(s);
    $("#debug").append(s);
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
