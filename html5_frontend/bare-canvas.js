var hm_globals = {
    PADDING: 16,    /*padding for the image as a whole. it should exceed FUZZ_RADIUS */
    FUZZ_RADIUS: 8, /*distance of points convolution */
    FUZZ_MAX: 15
};

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


function find_nice_shape_constant(k, peak, radius, offset){
    if (k >= 0){/*fools*/
        k = -0.5;
    }
    for (var i = 0; i < 100; i++){
        var a = parseInt(Math.exp(radius * k) * peak + offset);
        var outside = parseInt(Math.exp((radius + 1) * k) * peak + offset);
        if (a < 1){
            k *= 1 - Math.random() * 0.8;
        }
        else if (a > 1 || outside != 0){
            k /= 1 - Math.random() * 0.8;
        }
        else { //a == 1
            var b = parseInt(Math.exp((radius - 1) * k) * peak + offset);
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


/*XXX ignoring cases where CSS pixels are not device pixels */

function make_fuzz(radius){
    /* middle pixel + radius on either side */
    var size = 1 + 2 * radius;
    var middle = radius;
    var canvas = document.createElement("canvas");
    var helpers = document.getElementById("helpers");
    canvas.width = size;
    canvas.height = size;
    helpers.appendChild(canvas);
    var ctx = canvas.getContext("2d");
    var imgd = ctx.getImageData(0, 0, size, size);
    var pixels = imgd.data;
    var stride = size * 4;
    //var e = Math.E;
    /* find a good distribution for this size.
     * we want the 2 out pixels to be 1 and the inner pixel to be
     * hm_globals.FUZZ_MAX
     *  */
    var offset = 0.7;
    var peak = hm_globals.FUZZ_MAX;
    var k = find_nice_shape_constant(-0.5, peak, radius, offset);
    var s = "";
    for (var y = 0; y < size; y++){
        var dy2 = (y - radius) * (y - radius);
        var row = y * stride;
        for (var x = 0; x < size; x++){
            var dx2 = (x - radius) * (x - radius);
            /* aah, some formula so that
             * d == FUZZ_PIXELS -> difference of 1, just
             * d == 0           -> difference of say ~25
             * d == 1, 2, 3,... falls off sharply.
             *
             * presumably ~ e ^ -dk   (+ c ?)
             * */
            var a = parseInt(Math.exp(Math.sqrt(dx2 + dy2) * k) * peak + offset);
            var p = row + x * 4;
            s += a + " ";
            pixels[p] = 255;
            pixels[p + 1] = 255;
            pixels[p + 2] = 255;
            pixels[p + 3] = a;
        }
        s+="\n";
    }
    alert(k);
    ctx.putImageData(imgd, 0, 0);
    var img = new Image();
    img.src = canvas.toDataURL();
    return img;
}

function filter_rows(original, xmin, xmax, ymin, ymax){
    /*passing straight through is a common case*/
    if (xmin === undefined &&
        xmax === undefined &&
        ymin === undefined &&
        ymax === undefined){
        return original;
    }
    var rows = [];
    for (var i = 0; i < original.length; i++){
        var r = original[i];
        if ((xmin === undefined || xmin < r.key[0]) &&
            (xmax === undefined || xmax > r.key[0]) &&
            (ymin === undefined || ymin < r.key[1]) &&
            (ymax === undefined || ymax > r.key[1])){
            rows.push(r);
        }
    }
    return rows;
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
    var width = canvas.width - 2 * hm_globals.PADDING;
    var height = canvas.height - 2 * hm_globals.PADDING;
    var max_value = 0;
    var max_x = -1e999;
    var max_y = -1e999;
    var min_x =  1e999;
    var min_y =  1e999;
    var rows = filter_rows(data.rows);
    /*find the coordinate and value ranges */
    for (i = 0; i < rows.length; i++){
        var r = rows[i];
        max_value = Math.max(r.value, max_value);
        max_x = Math.max(r.key[0], max_x);
        max_y = Math.max(r.key[1], max_y);
        min_x = Math.min(r.key[0], min_x);
        min_y = Math.min(r.key[1], min_y);
    }

    var range_x = max_x - min_x;
    var range_y = max_y - min_y;
    var x_scale = width / range_x;
    var y_scale = height / range_y;

    var fuzz = make_fuzz(hm_globals.FUZZ_RADIUS);
    ctx.font = "10px Inconsolata";
    paste_fuzz(ctx, rows, fuzz, min_x, min_y, x_scale, y_scale);
    var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var height_map = img_data.data;
    for (i = 100000; i < 1000000; i++){
        height_map[i] = 255;
    }
}


function paste_fuzz(ctx, rows, img, min_x, min_y, x_scale, y_scale){
    if (img === undefined){
        img = new Image();
        img.src = "fuzz-19.png";
    }
    for (var i = 0; i < rows.length; i++){
        var r = rows[i];
        var x = hm_globals.PADDING + (r.key[0] - min_x) * x_scale;
        var y = hm_globals.PADDING + (r.key[1] - min_y) * y_scale;
        //ctx.putImageData(fuzz, x, y);
        ctx.drawImage(img, x, y);
    }
}

function blur_dots(ctx, rows, min_x, min_y, x_scale, y_scale){
    ctx.fillStyle = "rgba(255,255,255,1)";
    //ctx.fillStyle = "#888";
    ctx.shadowColor = "rgba(255,255,255,1)";
    ctx.shadowBlur = 6;
    //    ctx.globalAlpha = 0;
    for (i = 0; i < rows.length; i++){
        var r = rows[i];
        var x = hm_globals.PADDING + (r.key[0] - min_x) * x_scale;
        var y = hm_globals.PADDING + (r.key[1] - min_y) * y_scale;
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