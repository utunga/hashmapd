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
    //log(k, threshold, radius);
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
    //log(count, "expansions;", cols, "columns;", width, "width");
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
