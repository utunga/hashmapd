/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

/** make_fuzz_table_1d makes a one dimensional gaussian distribution */
function make_fuzz_table_1d(radius, k){
    var x;
    /* work out a table of 0-1 fuzz values */
    var table = [];
    var size = 2 * radius + 1;
    for (x = 0; x < size; x++){
        var dx2 = (x - radius) * (x - radius);
        //log(x, radius, dx2, k);
        table[x] = Math.exp(dx2 * k);
        if (table[x] < 0.001){
            table[x] = 0;
        }
    }
    return table;
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
function make_fuzz_array(points, radius, k,
                         img_width, img_height,
                         min_x, min_y,
                         x_scale, y_scale
                        ){
    var lut = make_fuzz_table_1d(radius, k);
    var len = lut.length;
    var x, y, i;

    /*we need 2 2D zeroed arrays.  If $const.ARRAY_FUZZ_TYPED_ARRAY is
     *set, Float32Array()s are used.  This is faster on some browsers
     *and slightly slower on others.  (It would be possible, with
     *marginal benefit, to test this dynamically and use the right
     *array in every browser).
     */
    var map = [];
    var row1, row;
    if ($const.ARRAY_FUZZ_TYPED_ARRAY){
        for (y = 0 ; y < img_height; y++){
            map[y] = new Float32Array(img_width);
        }
    }
    else {
        for (y = 0 ; y < img_height; y++){
            map[y] = row = [];
            for (x = 0; x < img_width; x++){
                row[x] = 0;
            }
        }
    }

    /* first pass: vertical spread from each point */
    var columns = {};
    /* extrema for simple pasting in-array */
    var max_oy = img_height - len;
    var min_oy = 0;
    for (i = 0; i < points.length; i++){
        var p = points[i];
        var py = parseInt((p[1] - min_y) * y_scale);
        var px = parseInt((p[0] - min_x) * x_scale);
        var pv = p[2];
        var oy = py - radius;
        var s = 0;
        var e = len;
        if (oy + e > img_height){
            e = img_height - oy;
        }
        if (oy < 0){
            s = -oy;
        }
        /* sparse columns.  */
        var col = columns[px];
        if (col == undefined){
            col = [];
            for (var j = 0; j < img_height; j++){
                col[j] = 0.0;
            }
            columns[px] = col;
        }
        for (y = s; y < e; y++){
            col[oy + y] += pv * lut[y];
        }
    }
    /* second pass: horizontal spread from all pixels */
    var count = 0;

    for (x in columns){
        for (y = 0; y < img_height; y++){
            row = map[y];
            var v = columns[x][y];
            if (v < 0.001){
                continue;
            }
            count ++;
            var ox = x - radius;
            var s = 0;
            var e = len;
            if (ox + e > img_width){
                e = img_width - ox;
            }
            if (ox < 0){
                s = -ox;
            }
            for (i = s; i < e; i++){
                row[ox + i] += v * lut[i];
            }
        }
    }
    log(count, "expansions");
    return map;
}

/** paste_fuzz_array gaussian kernel using arrays
 *
 * @param ctx a canvas 2d context to paint on
 * @param points the array of points
 * @param radius how far the influence of a point reaches
 * @param k a constant defining the shape of the gaussian
 *
 * <k> and <radius> should agree with each other: if <radius> is too
 * small for <k>, the image will show clipped square cliffs.  If it is
 * too large, it wastes time making infinitesimal changes.
*/
function paste_fuzz_array(ctx, map, radius, scale_radix, scale){
    var img_width = ctx.canvas.width;
    var img_height = ctx.canvas.height;
    var img_height = map.length;
    var img_width = map[0].length;
    var row;
    var x, y;
    /*find a good scale */
    var max_value = 0;
    for (y = 0; y < img_height; y++){
        row = map[y];
	for (x = 0; x < img_width; x++){
            if(max_value < row[x]){
                max_value = row[x];
            }
        }
    }
    /*do the map */
    var imgd = ctx.getImageData(0, 0, img_width, img_height);
    var pixels = imgd.data;
    var yend = img_height;// + radius;
    var xend = img_width;// + radius;
    var pix = 3;

    if (scale_radix == 0){
        if (scale == undefined){
            scale = $const.ARRAY_FUZZ_SCALE / max_value;
        }
        for (y = 0; y < yend; y++){
            row = map[y];
	    for (x = 0; x < xend; x++, pix += 4){
                pixels[pix] = parseInt(row[x] * scale);
            }
        }
    }
    else if (scale_radix < 0){
        //scale_radix is the exponent
        scale_radix = -scale_radix;
        if (scale == undefined){
            scale = $const.ARRAY_FUZZ_SCALE / (Math.pow(max_value, scale_radix) - 0.5);
        }
        for (y = radius; y < yend; y++){
            row = map[y];
	    for (x = radius; x < xend; x++, pix += 4){
                pixels[pix] = parseInt((Math.pow(row[x], scale_radix) - 0.5) * scale);
            }
        }
    }
    else{
        //scale_radix is the radix
        if (scale == undefined){
            scale = $const.ARRAY_FUZZ_SCALE / (Math.pow(scale_radix, max_value));
        }
        /* we need to offset the results a bit, because
         * {scale_radix ^ 0} == 1 which is multiplied by scale.
         *
         * So, to get that to zero, subtract scale, but to help
         * {scale_radix ^ 1} == scale_radix round to 1, we subtract
         * something a bit less.  Zero height pixels go to 0.95,
         * which is truncated to zero.
         */
        var offset = scale - 0.95;
        for (y = radius; y < yend; y++){
            row = map[y];
	    for (x = radius; x < xend; x++, pix += 4){
                pixels[pix] = parseInt(Math.pow(scale_radix, row[x]) *
                                       scale - offset);
            }
        }
    }
    ctx.putImageData(imgd, 0, 0);
    return scale;
}
