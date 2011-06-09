/* Copyright 2011 Hashmapd Ltd  <http://hashmapd.com>
 * written by Douglas Bagnall
 */

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

/** scaled_canvas helpfully makes a canvas the same shape as the main canvas.
 *
 * By default it is the same size also.
 *
 * @param p a size ratio, used to scale each dimension.
 *
 * In development it pastes the canvas onto the webpage/
 */
function scaled_canvas(p){
    p = p || 1;
    var w = ($hm.width + 2 * $hm.PADDING) * p;
    var h = ($hm.height + 2 * $hm.PADDING) * p;
    var canvas = new_canvas(w, h);
    document.getElementById("content").appendChild(canvas);
    return canvas;
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

/* Algorithm borrowed from John Barratt <http://www.langarson.com.au/>
 * as posted on Python Image-SIG in 2007.
 */

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

    scale = 1.0 / ($hm.HILL_SHADE_FLATNESS * scale);
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


function apply_density_map(ctx){
    var canvas2 = scaled_canvas();
    var ctx2 = canvas2.getContext("2d");
    var width = canvas2.width;
    var height = canvas2.height;
    ctx2.drawImage(ctx.canvas, 0, 0, width, height);

    var imgd = ctx2.getImageData(0, 0, width, height);
    var pixels = imgd.data;
    var height_pixels = $hm.height_canvas.getContext("2d").getImageData(0, 0, width, height).data;
    var map_pixels = $hm.canvas.getContext("2d").getImageData(0, 0, width, height).data;
    for (var i = 3, end = width * height * 4; i < end; i += 4){
        var x = pixels[i] * height_pixels[i];
        if(1){
            pixels[i - 3] = map_pixels[i - 2] * 2;
            pixels[i - 2] = map_pixels[i - 1] * 2;
            pixels[i - 1] = map_pixels[i - 3] * 2;
            //pixels[i] *= 0.65;
        }
        else{
            pixels[i] = 0;
        }
    }
    ctx2.putImageData(imgd, 0, 0);
    return canvas2;
}