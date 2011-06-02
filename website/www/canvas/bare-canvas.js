/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/* $hm holds global state.
 * Capitalised names are assumed to be constant (unnecessarily in some cases).
 */
var $hm = {
    DATA_URL: 'locations-15.json',
    //DATA_URL: 'http://hashmapd.couchone.com/frontend_dev/_design/user/_view/xy_coords?group=true',
    PADDING: 16,    /*padding for the image as a whole. it should exceed FUZZ_RADIUS */
    FUZZ_RADIUS: 10, /*distance of points convolution */
    FUZZ_MAX: 15,
    USING_QUAD_TREE: true
};

var HTTP_OK = 200;
var XML_HTTP_READY = 4;

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
    $.getJSON($hm.DATA_URL, function(data){
                  hm_on_data(canvas, data);
              });
}

/** make_colour_range utility
 *
 * @return a 256 long array of colours or gradients.
 */
function make_colour_range(){
    var colours = [];
    for (var i = 255; i >= 0; i--){
        var r = ((i >> 1) + 16);
        var g = i;
        var b = 0;
        colours.push('rgb(' + r + ',' + g + ',' + b + ')');
    }
    return colours;
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




function decode_and_filter_points(raw, xmin, xmax, ymin, ymax){
    var i, j;
    var points = [];
    if ($hm.USING_QUAD_TREE){
        for (i = 0; i < raw.length; i++){
            var r = raw[i];
            var coords = r.key;
            var x = 0;
            var y = 0;
            for (j = 0; j < coords.length; j++){
                var p = coords[j];
                x = (x << 1) | (p & 1);
                y = (y << 1) | (p >> 1);
            }
            points.push([x, y, r.value]);
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
    var ctx = canvas.getContext("2d");
    var width = canvas.width - 2 * $hm.PADDING;
    var height = canvas.height - 2 * $hm.PADDING;
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

    var range_x = max_x - min_x;
    var range_y = max_y - min_y;
    var x_scale = width / range_x;
    var y_scale = height / range_y;

    ctx.font = "10px Inconsolata";
    var fuzz = make_fuzz($hm.FUZZ_RADIUS);
    //alert(fuzz.complete);
    fuzz.onload = function(){
        paste_fuzz(ctx, points, fuzz, min_x, min_y, x_scale, y_scale);
        var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        var height_map = img_data.data;
        for (i = 100000; i < 1000000; i++){
            height_map[i] = 255;
        }
    };
}


function paste_fuzz(ctx, points, img, min_x, min_y, x_scale, y_scale){
    if (img === undefined){
        img = new Image();
        img.src = "fuzz-19.PNG";
    }
    for (var i = 0; i < points.length; i++){
        var r = points[i];
        var x = $hm.PADDING + (r[0] - min_x) * x_scale;
        var y = $hm.PADDING + (r[1] - min_y) * y_scale;
        //ctx.putImageData(fuzz, x, y);
        ctx.drawImage(img, x, y);
    }
}

function blur_dots(ctx, points, min_x, min_y, x_scale, y_scale){
    ctx.fillStyle = "rgba(255,255,255,1)";
    //ctx.fillStyle = "#888";
    ctx.shadowColor = "rgba(255,255,255,1)";
    ctx.shadowBlur = 6;
    //    ctx.globalAlpha = 0;
    for (i = 0; i < points.length; i++){
        var r = points[i];
        var x = $hm.PADDING + (r[0] - min_x) * x_scale;
        var y = $hm.PADDING + (r[1] - min_y) * y_scale;
        ctx.fillRect(x, y, 1, 1);
    }
}



function generate_height_map(){}

function add_label(ctx, text, x, y, text_fill, rect_fill, rect_w, rect_h){
    if (text_fill === undefined){
        text_fill = "#000";
    }
    ctx.fillStyle = text_fill;
    if (rect_w !== undefined && rect_h !== undefined){
        ctx.fillStyle = rect_fill;
        ctx.fillRect(x, y, rect_w, rect_h);
        ctx.fillStyle = text_fill;
    }
    ctx.fillText(text, x, y);
}
