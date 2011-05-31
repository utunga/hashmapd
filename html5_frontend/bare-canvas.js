var hm_globals = {
    PADDING: 16
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
    var rows = data.rows;
    var width = canvas.width - 2 * hm_globals.PADDING;
    var height = canvas.height - 2 * hm_globals.PADDING;
    var max_value = 0;
    var max_x = -1e999;
    var max_y = -1e999;
    var min_x =  1e999;
    var min_y =  1e999;
    /*find the coordinate and value ranges */
    for (i = 0; i < rows.length; i++){
        var r = rows[i];
        max_value = Math.max(r.value, max_value);
        max_x = Math.max(r.key[0], max_x);
        max_y = Math.max(r.key[1], max_y);
        min_x = Math.min(r.key[0], min_x);
        min_y = Math.min(r.key[1], min_y);
    }

    ctx.fillStyle = "#770";
    ctx.font = "10px Inconsolata";
    var range_x = max_x - min_x;
    var range_y = max_y - min_y;
    var x_scale = width / range_x;
    var y_scale = height / range_y;

    var labels = [];
    for (i = 0; i < rows.length; i++){
        var r = rows[i];
        var x = hm_globals.PADDING + r.key[0] * x_scale;
        var y = hm_globals.PADDING + r.key[1] * y_scale;
        ctx.fillRect(x, y, x_scale, y_scale);
        if (i % 100 == 0){
            labels.push([r, x, y]);
        }
    }

    for (i = 0; i < labels.length; i++){
        var d = labels[i];
        var key = d[0].key;

        ctx.fillStyle = "#f00";
        ctx.fillRect(d[1], d[2], x_scale, y_scale);
        ctx.fillStyle = "#000";
        ctx.fillText(("" + key[0]).substr(0,6) + "," + ("" + key[1]).substr(0, 6), d[1], d[2]);
    }
}
