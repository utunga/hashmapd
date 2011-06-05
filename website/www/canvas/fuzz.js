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
 * Look at the attributes of the $hm object that start with FUZZ_ for
 * documentation and useful values.
 *
 * The formula used is parseInt(Math.exp(k * d * d) + offset)
 *
 * @param densities make fuzz images with densities from 1 to <densities>
 * @param radius    size of guassian table. no image will exceed this.
 * @param k         negative inverse variance. closer to 0 is flatter.
 * @param offset    lifts floor in truncating (0.5 rounds, more to lengthen tails)
 * @param intensity level of fuzz at the center of density 1.
 *
 * @return an object referencing images that *will* contain fuzz when it it is ready.
 */

function make_fuzz(densities, radius, k, offset, intensity){
    var x, y;
    /* middle pixel + radius on either side */
    var tsize = 2 * radius + 1;
    var table = make_fuzz_table_2d(radius, k);

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
    images.table = table;
    images.table1d = centre_row;
    return images;
}


/** paste_fuzz_array gaussian kernel using arrays
 *
 * The 2d gaussian is decomposed into 2 1d gaussians.  THe first step
 * is quite quick because the number of accesses is exactly the
 * diameter times the number of points.  The second is slower because
 * the each pixel touched in the first round needs to be expanded.
 * Therefore it makes sense to perform the first round across the
 * grain, vertically.
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

function paste_fuzz_array(ctx, points, radius, k){
    var lut = make_fuzz_table_1d(radius, k);
    var len = lut.length;
    var radius = parseInt(len / 2);
    var img_width = ctx.canvas.width;
    var img_height = ctx.canvas.height;
    var array_width = img_width + lut.length;
    var array_height = img_height + lut.length;
    var x, y, i;
    var map1 = [];
    var map2 = [];
    if ($hm.ARRAY_FUZZ_TYPED_ARRAY){
        for (y = 0 ; y < array_height; y++){
            map1[y] = new Float32Array(array_width);
            map2[y] = new Float32Array(array_width);
        }
    }
    else {
        for (y = 0 ; y < array_height; y++){
            map1[y] = [];
            map2[y] = [];
            for (x = 0; x < array_width; x++){
                map1[y][x] = 0;
                map2[y][x] = 0;
            }
        }
    }

    /* first pass: vertical spread from each point */
    var offset = radius + $hm.PADDING;
    var counts = [];
    for (i = 0; i < points.length; i++){
        var p = points[i];
        var px = parseInt(offset + (p[0] - $hm.min_x) * $hm.x_scale);
        var py = parseInt(offset + (p[1] - $hm.min_y) * $hm.y_scale);
        var oy = py - radius;
        for (y = 0; y < len; y++){
            map1[oy + y][px] += lut[y];
        }
    }
    /* second pass: horizontal spread from all pixels */
    var count = 0;
    if(1){
        for (y = 0; y < array_height; y++){
            var row1 = map1[y];
            var row2 = map2[y];
	    for (x = 0; x < array_width; x++){
                var v = row1[x];
                if (v < 0.01){
                    continue;
                }
                count ++;
                var ox = x - radius;
                for (i = 0; i < len; i++){
                    row2[ox + i] += v * lut[i];
                }
            }
        }
    }
    else {
        for (y = radius; y < array_height - radius - 1; y++){
	    for (x = radius; x < array_width - radius - 1; x++){
                var acc = 0;
                for (var ix = 0; ix < len; ix++){
                    acc += map1[y][x + ix - radius] * lut[ix];
            }
                map2[y][x] = acc;
                count ++;
            }
        }
     }
    log(count, "expansions");
    /*find a good scale */
    var max_value = 0;
    for (y = 0; y < array_height; y++){
	for (x = 0; x < array_width; x++){
            if(max_value < map2[y][x]){
                max_value = map2[y][x];
            }
        }
    }
    /*do the map */
    var scale = 255.99 / Math.pow(1.27, max_value);
    //var scale = 255.99 / max_value;
    var imgd = ctx.getImageData(0, 0, img_width, img_height);
    var pixels = imgd.data;
    var row = 0;
    for (y = 0; y < img_height; y++){
        var row2 = map2[y + radius];
	for (x = 0; x < img_width; x++){
            var v = Math.pow(1.27, row2[x + radius]) * scale - 1;
            //var v = parseInt(row2[x + radius] * scale);
            if(v >= 1){
                pixels[row + x * 4] = 255;
                pixels[row + x * 4 + 3] = v;
            }
        }
        row += img_width * 4;
    }
    ctx.putImageData(imgd, 0, 0);
}