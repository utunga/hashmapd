var PADDING = 16;

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

/** hm_on_data is a callback from hm_draw_map.
 *
 * It coordinates the actual drawing.
 *
 * @param canvas the html5 canvas
 * @param data is parsed but otherwise unprocessed JSON data.
 */

function hm_on_data(canvas, data){
    var ctx = canvas.getContext("2d");
    var rows = data.rows;
    var scale = (canvas.width - 2 * PADDING) / (1 << rows[0].key.length);
    var max_value = 0;
    for (var j = 0; j < rows.length; j++){
        var r = rows[j];
        if (r.value > max_value){
            max_value = r.value;
        }
    }
    var colours = make_colour_range();
    var value_scale = 1.0 / max_value;
    var colour_scale = value_scale * (colours.length - 0.1);

    for (var j = 0; j < rows.length; j++){
        var r = rows[j];
        var coords = r.key;
        var x = 0;
        var y = 0;
        for (var i = 0; i < coords.length; i++){
           var p = coords[i];
            x = (x << 1) | (p & 1);
            y = (y << 1) | (p >> 1);
        }
        ctx.fillStyle = colours[parseInt(r.value * colour_scale)];
        ctx.fillRect(PADDING + x * scale, PADDING + y * scale,
                 scale, scale);
    }
}
