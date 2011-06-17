/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/** make_fuzz_table_1d makes a one dimensional gaussian distribution
 *
 * @param k is the fuzz constant (essentially -2 * variance)
 * @param threshold is the lowest number to include
 *
 * The distribution is clipped to the range where threshold is
 * exceeded.
 */
function make_fuzz_table_1d(k, threshold){
    if (! threshold)
        threshold = 0.01; /*there has to be something*/
    var half_table = [];
    for (var d = 0, f = threshold + 1; f > threshold; d++){
        f = Math.exp(d * d * k);
        half_table.push(f);
    }
    var table = half_table.slice(1);
    table.reverse();
    return table.concat(half_table);
}

function calc_fuzz_radius(k, threshold){
    if (! threshold)
        threshold = 0.01; /*there has to be something*/
    for (var d = 0, f = threshold + 1; f > threshold; d++){
        f = Math.exp(d * d * k);
    }
    return d;
}

/** make_fuzz_table_2d makes a two dimensional gaussian distribution */
function make_fuzz_table_2d(radius, k){
    var x, y;
    /* work out a table of 0-1 fuzz values */
    var table = [];
    /* middle pixel + radius on either side */
    var size = 2 * radius + 1;

    var height;
    var centre_y;
    for (y = 0; y < size; y++){
        table[y] = [];
        var ty = table[y];
        var dy2 = (y - radius) * (y - radius);
        for (x = 0; x < size; x++){
            var dx2 = (x - radius) * (x - radius);
            ty[x] = Math.exp((dx2 + dy2) * k);
        }
    }
    return table;
}


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
 * Look at the attributes of the $const object that start with FUZZ_ for
 * documentation and useful values.
 *
 * The formula used is parseInt(Math.exp(k * d * d) + offset)
 *
 * @param deferred  a $.Deferred object to fire when the images are ready
 * @param densities make fuzz images with densities from 1 to <densities>
 * @param radius    size of guassian table. no image will exceed this.
 * @param k         negative inverse variance. closer to 0 is flatter.
 * @param offset    lifts floor in truncating (0.5 rounds, more to lengthen tails)
 * @param intensity level of fuzz at the center of density 1.
 *
 * @return an object referencing images that *will* contain fuzz when it it is ready.
 */

function make_fuzz(deferred, densities, radius, k, offset, intensity){
    var x, y;
    /* middle pixel + radius on either side */
    var tsize = 2 * radius + 1;
    var table = make_fuzz_table_2d(radius, k);

    var centre_row = table[radius];
    var images = {loaded: 0,
                  ready: deferred
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
    images.table = table;
    images.table1d = centre_row;
    return images;
}

/**paste_fuzz puts image based fuzz on the image */

function paste_fuzz(ctx, points, images){
    var counts = [];
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        var x = (p[0] - $page.min_x) * $page.x_scale;
        var y = (p[1] - $page.min_y) * $page.y_scale;
        var count = p[2];
        counts[count] = (counts[count] || 0) + 1;
        var img;
        if (count <= $const.FUZZ_MAX_MULTIPLE){
            img = images[count];
        }
        else{
            /* XXX jump up to next scale */
            img = images[$const.FUZZ_MAX_MULTIPLE];
        }
        ctx.drawImage(img, x - img.width / 2, y - img.height / 2);
    }
}

/** zeroed_2d_array makes a 2d array suitable for floating point stuff
 *
 * If $const.ARRAY_FUZZ_TYPED_ARRAY is set, Float32Array()s are used.
 * This is faster on some browsers and slightly slower on others.  (It
 * would be possible, with marginal benefit, to test this dynamically
 * and use the right array in every browser).
 *
 * @param w
 * @param h
 *
 * @return an Array of either Arrays or Float32Arrays, filled with zeros.
 * */

function zeroed_2d_array(w, h){
    var x, y;
    var map = [];
    if ($const.ARRAY_FUZZ_TYPED_ARRAY){
        for (y = 0 ; y < h; y++){
            map[y] = new Float32Array(w);
        }
    }
    else {
        for (y = 0 ; y < h; y++){
            var row = map[y] = [];
            for (x = 0; x < w; x++){
                row[x] = 0.0;
            }
        }
    }
    return map;
}


/** make_fuzz_array uses a gaussian kernel function to blur points
 *
 * The 2d gaussian is decomposed into 2 1d gaussians.  The first step
 * is quite quick because the number of accesses is exactly the
 * diameter times the number of points.  The second is slower because
 * the each pixel touched in the first round needs to be expanded.
 * Therefore it makes sense to perform the first round across the
 * grain, vertically.
 *
 */
function make_fuzz_array(points, k, threshold,
                         width, height,
                         min_x, min_y,
                         x_scale, y_scale
                        ){
    var lut = make_fuzz_table_1d(k, threshold);
    var len = lut.length;
    var radius = parseInt(len / 2);
    log(k, threshold, radius);
    var x, y, i;
    var map = zeroed_2d_array(width, height);
    var row;

    /* first pass: vertical spread from each point */
    var columns = {};
    /* extrema for simple pasting in-array */
    var max_oy = height - len;
    var min_oy = 0;
    for (i = 0; i < points.length; i++){
        var p = points[i];
        var py = parseInt((p[1] - min_y) * y_scale);
        var px = parseInt((p[0] - min_x) * x_scale);
        var pv = p[2];
        var oy = py - radius;
        var s = 0;
        var e = len;
        if (oy + e > height){
            e = height - oy;
        }
        if (oy < 0){
            s = -oy;
        }
        /* sparse columns.  */
        var col = columns[px];
        if (col == undefined){
            col = [];
            for (var j = 0; j < height; j++){
                col[j] = 0.0;
            }
            columns[px] = col;
        }
        for (y = s; y < e; y++){
            col[oy + y] += pv * lut[y];
        }
    }
    /* second pass: horizontal spread from all pixels */
    var count = 0; /*counts additions */
    var cols = 0; /*for counting columns*/

    for (x in columns){
        for (y = 0; y < height; y++){
            row = map[y];
            var v = columns[x][y];
            if (v < 0.001){
                continue;
            }
            var ox = x - radius;
            var s = 0;
            var e = len;
            if (ox + e > width){
                e = width - ox;
            }
            if (ox < 0){
                s = -ox;
            }
            for (i = s; i < e; i++){
                row[ox + i] += v * lut[i];
            }
            count += e - s;
        }
        cols += 1;
    }
    log(count, "expansions;", cols, "columns;", width, "width");
    return map;
}

/** paste_fuzz_array gaussian kernel using arrays
 *
 * @param ctx a canvas 2d context to paint on
 * @param map a 2d array of floating point values
 * @param scale_args array determining scaling of map values
 * @param max_value considered highest value in map (undefined for auto)
 *
 * @return the given or discovered max_value.
*/
function paste_fuzz_array(ctx, map, scale_args, max_value){
    var height = map.length;
    var width = map[0].length;
    var row;
    var x, y;

    if (max_value === undefined){
        max_value = 0;
        /*find the maximum to calculate a good scale */
        for (y = 0; y < height; y++){
            row = map[y];
	    for (x = 0; x < width; x++){
                if(max_value < row[x]){
                    max_value = row[x];
                }
            }
        }
    }
    /*do the map */
    var imgd = ctx.getImageData(0, 0, width, height);
    var pixels = imgd.data;
    var pix = 3;
    /*concat rather than unshift, lest scale_args grow over time. */
    var args = [max_value].concat(scale_args);
    var lut = get_fuzz_scale_lut.apply(undefined, args);
    var scale = lut.scale;
    for (y = 0; y < height; y++){
        row = map[y];
	for (x = 0; x < width; x++, pix += 4){
            pixels[pix] = lut[parseInt(row[x] * lut.scale)];
        }
    }
    ctx.putImageData(imgd, 0, 0);
    return max_value;
}

function get_fuzz_scale_lut(max_value, mode){
    var i;
    var lut = [];
    var len = $const.ARRAY_FUZZ_LUT_LENGTH;
    var scale = (len - 0.1) / max_value;
    lut.scale = scale;
    var f;
    var max_out = $const.ARRAY_FUZZ_SCALE;

    if (mode == 'linear'){
        f = function(i){
            return parseInt(i * max_out / len);
        };
    }
    else if (mode == 'base'){
        //radix is the exponent
        var radix = arguments[2];
        var s = max_out / Math.pow(len, radix);
        f = function(i){
            return parseInt((Math.pow(i, radix) - 0.5) * s);
        };
    }
    else if (mode == 'clipped_gaussian'){
        /* clip a piece out of the normal curve.  The desired
         * characteristics are: a flat start, a definite knee, and a
         * limit to the eventual slope.
         */
        var rl = arguments[2];
        var rh = arguments[3];
        var range = Math.abs(rl - rh);
        var top = Math.exp(rh);
        var bottom = Math.exp(rl);
        var s = max_out / (top - bottom);

        f = function(i){
            var p = rl + i / len  * range;
            return parseInt((Math.exp(p)  - bottom) * s);
        };
    }
    else{
        log('unknown mode in get_fuzz_scale_lut:', mode);
    }

    for (i = 0; i < len; i++){
        lut.push(f(i));
    }
    /*add on a whole lot of head room (flat). Clipping is better than NaN-ing*/
    var k = f(len - 1);
    for (i = 0; i < len; i++){
        lut.push(k);
    }

    //alert(lut);
    return lut;
}
