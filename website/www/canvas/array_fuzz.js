/** 2d decomposition of gaussian

*/

function paste_fuzz_array(ctx, points, images){
    var lut = $hm.hill_fuzz.table1d;
    var len = lut.length;
    var radius = parseInt(len / 2);
    var img_width = ctx.canvas.width;
    var img_height = ctx.canvas.height;
    var array_width = img_width + lut.length;
    var array_height = img_height + lut.length;
    var x, y;
    var map1 = [];
    var map2 = [];
    for (y = 0 ; y < array_height; y++){
        map1[y] = [];
        map2[y] = [];
        for (x = 0; x < array_width; x++){
            map1[y][x] = 0;
            map2[y][x] = 0;
        }
    }
    /* first pass: vertical spread from each point */
    var offset = radius + $hm.PADDING;
    var counts = [];
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        var px = parseInt(offset + (p[0] - $hm.min_x) * $hm.x_scale);
        var py = parseInt(offset + (p[1] - $hm.min_y) * $hm.y_scale);
        var oy = py - radius;
        for (y = 0; y < len; y++){
            map1[oy + y][px] += lut[y];
        }
    }
    /* second pass: horizontal spread from all pixels */

    if(1){
        for (y = 0; y < array_height; y++){
	    for (x = 0; x < array_width; x++){
                var v = map1[y][x];
                if (v < 0.00001){
                    continue;
                }
                var ox = x - radius;
                for (var i = 0; i < len; i++){
                    map2[y][ox + i] += v * lut[i];
                }
            }
        }
    }
    else {
        for (y = radius; y < array_height - radius - 1; y++){
	    for (x = radius; x < array_width - radius - 1; x++){
                var acc = 0;
                for (var ix = 0; ix < len; ix++){
                    acc += map1[y][x + ix] * lut[ix];
            }
                map2[y][x] = acc;
            }
        }
     }

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
    var imgd = ctx.getImageData(0, 0, img_width, img_height);
    var pixels = imgd.data;
    var row = 0;
    for (y = 0; y < img_height; y++){
	for (x = 0; x < img_width; x++){
            var v = Math.pow(1.27, map2[y + radius][x + radius]) * scale -1;
            if(v > 0){
                pixels[row + x * 4] = 255;
                pixels[row + x * 4 + 3] = v;
            }
        }
        row += img_width * 4;
    }
    ctx.putImageData(imgd, 0, 0);
}
