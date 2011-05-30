var DATA_URL = 'locations.json';
var HTTP_OK = 200;
var XML_HTTP_READY = 4;
var PADDING = 16;

/** hm_on_data is a callback from hm_draw_map.
 *
 * It coordinates the actual drawing.
 *
 * @param canvas the html5 canvas
 * @param data is parsed but otherwise unprocessed JSON data.
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
            /* start from other end */
            var p = coords[coords.length - i - 1];
            x += (p & 1) * (1 << i);
            y += (p >> 1) * (1 << i);
        }
        ctx.globalAlpha = r.value * value_scale;
        ctx.fillStyle = colours[parseInt(r.value * colour_scale)];
        ctx.beginPath();
        ctx.arc(PADDING + x * scale, PADDING + y * scale,
                scale * 2, 0, 6.3);
        ctx.fill();
    }
}

/** hm_draw_map is the main entrance point.
 *
 * Nothing happens until the json is loaded.
 *
 * @param canvas is the html5 canvas element to draw on
 */


function hm_draw_map(canvas){
    var req = new XMLHttpRequest();
    req.open("GET", DATA_URL, true);
    req.onreadystatechange = function(){
        /*XXX could arguably begin drawing before data is finished */
        if (req.readyState == XML_HTTP_READY) {
            var data = JSON.parse(req.responseText);
            hm_on_data(canvas, data);
        }
    };
    req.send(null);
}
