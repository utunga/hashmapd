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
    canvas.draggable = false;
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
function scaled_canvas(p, id){
    p = p || 1;
    var w = $const.width * p;
    var h = $const.height * p;
    var canvas = new_canvas(w, h, id);
    if ($const.DEBUG){
        document.getElementById("content").appendChild(canvas);
    }
    return canvas;
}

/* named_canvas will get or create a canvas in $page.canvases
 *
 * @param name is the name for the canvas.
 * @param blank clears or (if css colour string) the canvas
 * @param p sets the size (proportional to map size).
 */
function named_canvas(name, blank, p){
    var canvas = $page.canvases[name];
    if (canvas === undefined){
        canvas = scaled_canvas(p, "canvas-" + name);
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

/*overlay_canvas calls named_canvas, and overlays it on the main canvas
 *
 * Sorry for all the concentric functions.
 *
 * @param name is a name for the canvas
 * @param hidden flags whether the canvas is visible to stgart with.
 * @return the canvas
 */

function overlay_canvas(name, hidden, blank){
    var canvas = named_canvas(name, blank);
    overlay(canvas, hidden);
    return canvas;
}

function overlay(canvas, hidden){
    $("#map-div").append(canvas);
    $(canvas).css("position", "absolute");
    var vis = hidden ? 'hidden' : 'visible';
    $(canvas).css("visibility", vis);
    $(canvas).addClass("overlay").offset($($page.canvas).offset());
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
    if (1){
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

function add_label(ctx, text, x, y, size, colour, shadow, angle){
    //log.apply(undefined, arguments);
    if (colour === undefined){
        colour = "#000";
    }
    if (angle){
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        x = 0; y = 0;
    }
    if (shadow && size > 8){
        ctx.shadowColor = shadow;
        ctx.shadowBlur = size * 0.125;
        ctx.shadowOffsetX = 0.5 + size * 0.125;
        ctx.shadowOffsetY = 0.5 + size * 0.125;
    }
    //ctx.globalAlpha = "0.6";
    ctx.textAlign = "center";
    ctx.font = size + "px sans-serif";
    ctx.fillStyle = colour;
    ctx.fillText(text, x, y);

    if (angle){
        ctx.restore();
    }
}


function apply_density_map(src_ctx){
    var canvas = named_canvas("density_overlay", true);
    var ctx = canvas.getContext("2d");
    var width = canvas.width;
    var height = canvas.height;
    /* paste the raw image on the canvas */
    ctx.drawImage(src_ctx.canvas, 0, 0, width, height);
    var imgd = ctx.getImageData(0, 0, width, height);
    var pixels = imgd.data;
    var map_pixels = $page.canvas.getContext("2d").getImageData(0, 0, width, height).data;
    /* two possible height references, depending on whether zoom is zero */
    var hctx;
    if ($state.zoom == 0){
        hctx = $page.height_canvas.getContext("2d");
    }
    else {
        hctx = named_canvas("zoomed_height_map").getContext("2d");
    }
    var height_pixels = hctx.getImageData(0, 0, width, height).data;

    var func = {
        colour_cycle: colour_cycle,
        grey_outside: grey_outside
    }[$const.DENSITY_MAP_STYLE];

    func(pixels, map_pixels, height_pixels);

    ctx.putImageData(imgd, 0, 0);
    return canvas;
}

function colour_cycle(pix, map_pix, height_pix){
    for (var i = 3, end = pix.length; i < end; i += 4){
        var x = pix[i] * height_pix[i];
        if(x){
            pix[i - 3] = (map_pix[i - 2] * 2 - map_pix[i - 1]);
            pix[i - 2] = (map_pix[i - 1] * 2 - map_pix[i - 3]);
            pix[i - 1] = (map_pix[i - 3] * 2 - map_pix[i - 2]);
        }
        else{
            pix[i] = 0;
        }
    }
    return pix;
}

function grey_outside(pix, map_pix, height_pix){
    for (var i = 3, end = pix.length; i < end; i += 4){
        pix[i] = 255 - pix[i];
        var r = map_pix[i - 3];
        var g = map_pix[i - 2];
        var b = map_pix[i - 1];
        var grey = (r * 2 + g * 5 + b) >> 3;
        pix[i - 3] = grey;
        pix[i - 2] = grey;
        pix[i - 1] = grey;
    }
    return pix;
}




function paint_density_array(ctx, points){
    //token_ctx.fillStyle = "#f00";
    //token_ctx.fillRect(0, 0, token_ctx.canvas.width, token_ctx.canvas.height);
    var map = make_fuzz_array(points,
        $const.ARRAY_FUZZ_DENSITY_CONSTANT,
        $const.ARRAY_FUZZ_DENSITY_THRESHOLD,
        ctx.canvas.width, ctx.canvas.height,
        $page.min_x, $page.min_y,
        $page.x_scale  * 0.25, $page.y_scale * 0.25);
    paste_fuzz_array(ctx, map,
                     $const.ARRAY_FUZZ_DENSITY_SCALE_ARGS
                    );
}


/** zoom_in puts a bit of the src canvas all over the dest canvas,
 *
 *  This uses the built-in canvas blit, and is very very quick.
 *
 * @param src a canvas or 2d context to copy from
 * @param dest a canvas or 2d context to paste to
 * @param x
 * @param y
 * @param w
 * @param h define the rectangle to copy from.
 *
 */
function zoom_in(src, dest, x, y, w, h){
    if (dest.getContext){
        dest = dest.getContext("2d");
    }
    if (src.canvas){
        src = src.canvas;
    }
    /* maybe the correct thing to do is:
     dest.drawImage(src, parseInt(x), parseInt(y), parseInt(w), parseInt(h),
     0, 0, dest.canvas.width, dest.canvas.height);
     }
     */
    dest.drawImage(src, x, y, w, h, 0, 0, dest.canvas.width, dest.canvas.height);
}
