/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/* $hm holds global state.
 * Capitalised names are assumed to be constant (unnecessarily in some cases).
 */
var $hm = {
    DATA_URL: 'locations-15.json',
    TOKEN_DENSITY_URL: 'token_density-8.json',
    LABELS_URL: 'tokens-7.json',
    //DATA_URL: 'http://hashmapd.couchone.com/frontend_dev/_design/user/_view/xy_coords?group=true',
    PADDING: 16,    /*padding for the image as a whole. it should exceed FUZZ_RADIUS */
    FUZZ_RADIUS: 9, /*distance of points convolution */
    FUZZ_MAX: 15,
    USING_QUAD_TREE: true,
    QUAD_TREE_COORDS: 15,
    mapping_done: false, /*set to true when range, origin and scale are decided */
    landscape_done: false, /*set to true when finished drawing landscape */
    canvas: undefined,  /* a reference to the main canvas gets put here */
    /* convert data coordinates to canvas coordinates */
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
 * Nothing happens until the json is loaded, then the hm_on_data
 * function is called with the canvas reference and JSON
 * data. hm_on_data is differently defined for processing and bare
 * canvas implementations..
 *
 * @param canvas is the html5 canvas element to draw on
 */

function hm_draw_map(canvas){
    $hm.canvas = canvas;
    $hm.width = canvas.width - 2 * $hm.PADDING;
    $hm.height = canvas.height - 2 * $hm.PADDING;

    $.getJSON($hm.DATA_URL, function(data){
                  hm_on_data(canvas, data);
              });
    /*
    $.getJSON($hm.LABELS_URL, function(data){
                  hm_on_labels(canvas, data);
              });}
     */
    $.getJSON($hm.TOKEN_DENSITY_URL, function(data){
                  hm_on_token_density(canvas, data);
              });
}

function find_nice_shape_constant(k, peak, radius, offset, concentration){
    if (k >= 0){/*fools, including myself*/
        k = -0.5;
    }
    var max_spill = 0.67;
    for (var i = 0; i < 200; i++){
        var a = parseInt(Math.exp(radius * k) * peak + offset);
        var outside = parseInt(Math.exp((radius + max_spill) * k) * peak + offset);
        if (a < 1){
            k *= 1 - Math.random() * 0.6;
        }
        else if (a > 1 || outside != 0){
            k /= 1 - Math.random() * 0.6;
        }
        else {
            /* a == 1, and outside == 0.
             * Now, check concentration.
             *
             */
            var b = parseInt(Math.exp((radius * (1 - concentration)) * k) * peak + offset);
            if (b < 1){
                k *= 1 - Math.random() * 0.4;
            }
            else if (b > 1){
                k /= 1 - Math.random() * 0.4;
            }
            else {
                return k;
            }
        }
    }
    /*give up*/
    return k;
}

function new_canvas(width, height, id){
    var canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    if (id){
        canvas.id = id;
    }
    return canvas;
}

/*XXX ignoring cases where CSS pixels are not device pixels */

/** Make_fuzz makes a little fuzzy circle image for point convolution
 *
 * The returned image is not necessarily ready when the function
 * returns: you must make use of it in an .onload() handler (or poll
 * the .complete attribute).
 *
 * The fuzz is approximately gaussian and carried in the images alpha
 * channel.  All values are 8-bit integers less than <peak>.  The
 * shape is tuned stochastically until the pixels at <radius> and at
 * (1 - <concetnration>) * <radius> have alpha 1, while those a bit
 * beyond <radius> have alpha 0. The image is sized <radius> * 2 + 1.
 *
 * If you ignore <concentration> and <floor>, sensible (or at least
 * frequently successful) defaults will be chosen.
 *
 * The formula used is Math.exp(k * d) + <floor>
 *
 * where k is fiddled until both <radius> and <radius> * (1 -
 * <concentration>) are 1.  Thus increasing <concentration> flattens
 * the outside and steepens the centre.
 *
 * @param radius is the distance
 * @param peak is the alpha for the centre pixel
 * @param concentration is a sort of inverse variance. try 0.25.
 * @param floor positive values lift the whole curve, creating long tails.
 *
 * @return an Image object that *will* contain the fuzz when it it is ready.
 */

function make_fuzz(radius, peak, concentration, floor){
    peak = (peak === undefined) ? $hm.FUZZ_MAX : peak;
    concentration = (concentration === undefined) ? 0.25 : concentration;
    var offset = (floor === undefined) ? 0.7 : floor + 0.5;
    /* middle pixel + radius on either side */
    var size = 1 + 2 * radius;
    var canvas = new_canvas(size, size);
    var helpers = document.getElementById("helpers");
    helpers.appendChild(canvas);
    var ctx = canvas.getContext("2d");
    var imgd = ctx.getImageData(0, 0, size, size);
    var pixels = imgd.data;
    var stride = size * 4;
    var k = find_nice_shape_constant(-0.5, peak, radius, offset, concentration);
    var s = "";
    for (var y = 0; y < size; y++){
        var dy2 = (y - radius) * (y - radius);
        var row = y * stride;
        for (var x = 0; x < size; x++){
            var dx2 = (x - radius) * (x - radius);
            var a = parseInt(Math.exp(Math.sqrt(dx2 + dy2) * k) * peak + offset);
            var p = row + x * 4;
            s += a + " ";
            pixels[p] = 255;
            pixels[p + 3] = a;
        }
        s+="\n";
    }
    ctx.putImageData(imgd, 0, 0);
    var img = new Image();
    img.src = canvas.toDataURL();
    return img;
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
                //alert(j + " " + coords[j]);
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

function hm_on_data(canvas, data){
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

    $hm.range_x = max_x - min_x;
    $hm.range_y = max_y - min_y;
    $hm.x_scale = width / $hm.range_x;
    $hm.y_scale = height / $hm.range_y;
    $hm.min_x = min_x;
    $hm.min_y = min_y;
    $hm.max_x = max_x;
    $hm.max_y = max_y;
    $hm.mapping_done = true;
    var ctx = canvas.getContext("2d");
    var fuzz = make_fuzz($hm.FUZZ_RADIUS);
    var fuzz_canvas = new_canvas(canvas.width, canvas.height);
    $(fuzz_canvas).insertAfter(canvas);
    var fuzz_ctx = fuzz_canvas.getContext("2d");
    fuzz.onload = function(){ /* fuzz is async */
        paste_fuzz(fuzz_ctx, points, fuzz);
        hillshading(fuzz_ctx, ctx, 1, Math.PI * 1 / 4, Math.PI / 4);
        $hm.landscape_done = true;
        //alert($hm);
    };
}


function paste_fuzz(ctx, points, img){
    for (var i = 0; i < points.length; i++){
        var r = points[i];
        var x = $hm.PADDING + (r[0] - $hm.min_x) * $hm.x_scale;
        var y = $hm.PADDING + (r[1] - $hm.min_y) * $hm.y_scale;
        ctx.drawImage(img, x - img.width / 2, y - img.height / 2);
    }
}

function hillshading(map_ctx, target_ctx, scale, angle, alt){
    var canvas = target_ctx.canvas;
    var width = canvas.width;
    var height = canvas.height;
    var map_imgd = map_ctx.getImageData(0, 0, width, height);
    var map_pixels = map_imgd.data;
    var target_imgd = target_ctx.getImageData(0, 0, width, height);
    var target_pixels = target_imgd.data;

    scale = 1.0 / (8.0 * scale);
    var sin_alt = Math.sin(alt);
    var cos_alt = Math.cos(alt);
    var perpendicular = angle - Math.PI / 2;
    var stride = height * 4;
    var colours = make_colour_range_mountains(115);
    var row = stride; /*start on row 1, not row 0 */
    for (var y = 1, yend = height - 1; y < yend; y++){
        for (var x = 4 + 3, xend = stride - 4; x < xend; x += 4){
            var a = row + x;
            var cc = map_pixels[a];
            if (cc < 1){
                target_pixels[a - 3] = 147;
                target_pixels[a - 2] = 187;
                target_pixels[a - 1] = 189;
                target_pixels[a] = 255;
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

            /* Slope */
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
        [ 95,  90,  20,   0],
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


function hm_on_token_density(canvas, data){
    var i;
    var points = decode_and_filter_points(data.rows,
                                          $hm.min_x, $hm.max_x,
                                          $hm.min_y, $hm.max_y);

    var max_freq = 0;
    for (var i = 0; i < points.length; i++){
        var freq = points[i][2];
        max_freq = Math.max(freq, max_freq);
    }
    var scale = 14 / (max_freq * max_freq);
    var ctx = canvas.getContext("2d");

    function add_labels(){
        for (var i = 0; i < points.length; i++){
            var p = points[i];
            var x = $hm.PADDING + (p[0] - $hm.min_x) * $hm.x_scale;
            var y = $hm.PADDING + (p[1] - $hm.min_y) * $hm.y_scale;
            var text = p[3];
            var n = p[2];
            var size = n * n * scale;
            add_label(ctx, text, x, y, size, "#000", "#fff");
        }
    }
    wait_for_flag("landscape_done", add_labels);
}



/*don't do too much until the drawing is done.*/

function hm_on_labels(canvas, data){
    var i;
    //alert(data.rows);
    var points = decode_and_filter_points(data.rows,
                                          $hm.min_x, $hm.max_x,
                                          $hm.min_y, $hm.max_y);
    var labels = [];
    var max_freq = 0;
    for (var i = 0; i < points.length; i++){
        var freq = points[i][2][0][1];
        max_freq = Math.max(freq, max_freq);
    }
    var scale = 14 / (max_freq * max_freq);
    var ctx = canvas.getContext("2d");

    function add_labels(){
        for (var i = 0; i < points.length; i++){
            var p = points[i];
            var x = $hm.PADDING + (p[0] - $hm.min_x) * $hm.x_scale;
            var y = $hm.PADDING + (p[1] - $hm.min_y) * $hm.y_scale;
            var text = p[2][0][0];
            var n = p[2][0][1];
            var size = n * n * scale;
            add_label(ctx, text, x, y, size, "#000", "#fff");
        }
        //alert(points[3].toSource());
    }
    wait_for_flag("landscape_done", add_labels);
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
