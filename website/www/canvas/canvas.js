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
    var w = $const.width * p;
    var h = $const.height * p;
    var canvas = new_canvas(w, h);
    document.getElementById("content").appendChild(canvas);
    return canvas;
}

/* get a particular canvas that is a member of $page, or if it doesn't
 * exist, make it up and store it */
function named_canvas(name, blank, p){
    var canvas = $page.canvases[name];
    if (canvas === undefined){
        canvas = scaled_canvas(p);
        $page.canvases[name] = canvas;
    }
    if (blank){
        var ctx = canvas.getContext("2d");
        if (typeof(blank) == 'string'){
            /* this doesn't work on all browsers */
            ctx.fillStyle = blank;
            ctx.globalCompositeOperation = 'copy';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
        else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
    return canvas;
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
    $timestamp("start lut");
    var padding = 1;
    if (0){
        var _hill_slope = _hill_slope_wide_big;
        padding = 2;
        scale /= 4;

    }
    else if (0){
        var _hill_slope = _hill_slope_wide_small;
        padding = 2;
        scale /= 4;

    }
    else if (1){
        var _hill_slope = _hill_slope_big;
    }
    else {
        scale /= 4;
        var _hill_slope = _hill_slope_small;
    }

    var lut = hillshading_lut(scale, angle, alt);
    //log(lut);
    $timestamp("made lut");
    var lut_offset = $const.HILL_LUT_CENTRE;
    /*colour in the sea in one go */
    target_ctx.fillStyle = "rgb(147,187,189)";
    target_ctx.fillRect(0, 0, width, height);
    var target_imgd = target_ctx.getImageData(0, 0, width, height);
    var target_pixels = target_imgd.data;
    var stride = width * 4;
    var colours = make_colour_range_mountains(135);
    var row = stride * padding; /*start on row 1, not row 0 */
    for (var y = padding, yend = height - padding; y < yend; y++){
        for (var x = padding * 4 + 3, xend = stride - padding * 4; x < xend; x += 4){
            var a = row + x;
            var cc = map_pixels[a];
            if (cc < 1){
                continue;
            }
            var c = _hill_slope(map_pixels, a, stride, lut, lut_offset);
            var colour = colours[cc];
            if (cc == 1){ /* the sea shore has less shadow */
                c = (0.5 + c) / 2;
            }
            target_pixels[a - 3] = colour[0] + 120 * c;
            target_pixels[a - 2] = colour[1] + 120 * c;
            target_pixels[a - 1] = colour[2] + 120 * c;
            target_pixels[a] = 255;
        }
        row += stride;
    }
    target_ctx.putImageData(target_imgd, 0, 0);
}

function _hill_slope_small(map_pixels, a, stride, lut, lut_offset){
    var tc = map_pixels[a - stride];
    var cl = map_pixels[a - 4];
    var cr = map_pixels[a + 4];
    var bc = map_pixels[a + stride];
    var _dx = ((cl) - (cr));
    var _dy = ((bc) - (tc));

    return lut[lut_offset + _dy][lut_offset + _dx];
}

function _hill_slope_big(map_pixels, a, stride, lut, lut_offset){
    var tc = map_pixels[a - stride];
    var tl = map_pixels[a - stride - 4];
    var tr = map_pixels[a - stride + 4];
    var cl = map_pixels[a - 4];
    var cr = map_pixels[a + 4];
    var bc = map_pixels[a + stride];
    var bl = map_pixels[a + stride - 4];
    var br = map_pixels[a + stride + 4];

    var _dx = ((tl + 2 * cl + bl) - (tr + 2 * cr + br));
    var _dy = ((bl + 2 * bc + br) - (tl + 2 * tc + tr));
    return lut[lut_offset + _dy][lut_offset + _dx];
}

function _hill_slope_wide_small(map_pixels, a, stride, lut, lut_offset){
    stride *= 2;
    var tc = map_pixels[a - stride];
    var cl = map_pixels[a - 8];
    var cr = map_pixels[a + 8];
    var bc = map_pixels[a + stride];
    var _dx = ((cl) - (cr));
    var _dy = ((bc) - (tc));

    return lut[lut_offset + _dy][lut_offset + _dx];
}

function _hill_slope_wide_big(map_pixels, a, stride, lut, lut_offset){
    stride *= 2;
    var tc = map_pixels[a - stride];
    var tl = map_pixels[a - stride - 8];
    var tr = map_pixels[a - stride + 8];
    var cl = map_pixels[a - 8];
    var cr = map_pixels[a + 8];
    var bc = map_pixels[a + stride];
    var bl = map_pixels[a + stride - 8];
    var br = map_pixels[a + stride + 8];

    var _dx = ((tl + 2 * cl + bl) - (tr + 2 * cr + br));
    var _dy = ((bl + 2 * bc + br) - (tl + 2 * tc + tr));
    _dx = parseInt(0.25 * _dx);
    _dy = parseInt(0.25 * _dy);
    if (isNaN(_dx + _dy)){
        log(a, lut_offset, _dx, _dy, tl, tc, tr, cl,  cr, bl, bc, br);
    }
    return lut[lut_offset + _dy][lut_offset + _dx];
}



/*make hillshading LUT*/
function hillshading_lut(scale, angle, alt){
    var key = "lut_" + scale + "_" + angle + "_" + alt;
    if ($page.hillshade_luts[key] !== undefined){
        return $page.hillshade_luts[key];
    }

    scale = 1.0 / ($const.HILL_SHADE_FLATNESS * scale);
    var sin_alt = Math.sin(alt);
    var cos_alt = Math.cos(alt);
    var perpendicular = angle - Math.PI / 2;
    var dx, dy;
    var table = [];
    var lut_centre = $const.HILL_LUT_CENTRE;
    for (dy = -lut_centre; dy <= lut_centre; dy++){
        //var row = [];
        var row = new Float32Array(2 * lut_centre + 1);
        table[dy + lut_centre] = row;
        for (dx = -lut_centre; dx <= lut_centre; dx++){
            /* Slope -- there may be a quicker way of calculating sin/cos */
            var c = _hillshade(dx, dy, scale, sin_alt, cos_alt, perpendicular);
            row[lut_centre + dx] = c;
        }
    }
    $page.hillshade_luts[key] = table;
    return table;
}

function _hillshade(_dx, _dy, scale, sin_alt, cos_alt, perpendicular){
    var dx = _dx * scale;
    var dy = _dy * scale;
    var slope = Math.PI / 2 - Math.atan(Math.sqrt(dx * dx + dy * dy));

    var sin_slope = Math.sin(slope);
    var cos_slope = Math.cos(slope);

    /* Aspect */
    var aspect = Math.atan2(dx, dy);

    /* Shade value */
    var c = Math.max(sin_alt * sin_slope + cos_alt * cos_slope *
                     Math.cos(perpendicular - aspect), 0);
    return c;
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
        [ 60, 105,  10,   8],
        [ 50,  70,  10,  60],
        [ 55,  50,   0, 100],
        [ 55,  55,  45, 150],
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
    var height_pixels = $page.height_canvas.getContext("2d").getImageData(0, 0, width, height).data;
    var map_pixels = $page.canvas.getContext("2d").getImageData(0, 0, width, height).data;
    for (var i = 3, end = width * height * 4; i < end; i += 4){
        var x = pixels[i] * height_pixels[i];
        if(x){
            pixels[i - 3] = (map_pixels[i - 2] * 2 - map_pixels[i - 1]);
            pixels[i - 2] = (map_pixels[i - 1] * 2 - map_pixels[i - 3]);
            pixels[i - 1] = (map_pixels[i - 3] * 2 - map_pixels[i - 2]);
            //pixels[i] *= 0.65;
        }
        else{
            pixels[i] = 0;
        }
    }
    ctx2.putImageData(imgd, 0, 0);
    return canvas2;
}

function paint_density_array(ctx, points){
    //token_ctx.fillStyle = "#f00";
    //token_ctx.fillRect(0, 0, token_ctx.canvas.width, token_ctx.canvas.height);
    var map = make_fuzz_array(points,
        $const.ARRAY_FUZZ_DENSITY_RADIUS,
        $const.ARRAY_FUZZ_DENSITY_CONSTANT,
        ctx.canvas.width, ctx.canvas.height,
        $page.min_x, $page.min_y,
        $page.x_scale  * 0.25, $page.y_scale * 0.25);
    paste_fuzz_array(ctx, map,
                     $const.ARRAY_FUZZ_DENSITY_RADIUS,
                     $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS
                    );
}


/** zoom_in puts a bit of the src canvas all over the dest canvas */
function zoom_in(src, dest, x, y, w, h){
    $timestamp("zooming");
    if (dest.getContext){
        dest = dest.getContext("2d");
    }
    if (src.canvas){
        src = src.canvas;
    }
    if(1){
        /* maybe the correct thing to do is:
            dest.drawImage(src, parseInt(x), parseInt(y), parseInt(w), parseInt(h),
                           0, 0, dest.canvas.width, dest.canvas.height);
                                   }
         */
        dest.drawImage(src, x, y, w, h, 0, 0, dest.canvas.width, dest.canvas.height);
    }
    else {
        var sx = dest.canvas.width / w;
        var sy = dest.canvas.width / h;

        dest.save();
        dest.scale(sx, sy);
        dest.drawImage(src, x, y, w, h, 0, 0, w, h);
        dest.restore();
    }
    $timestamp("end zoom");
}
