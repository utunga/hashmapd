var DATA_URL = 'locations.json';
var HTTP_OK = 200;
var XML_HTTP_READY = 4;

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
        var r = ((i >> 1) + 64).toString(16);
        var g = i.toString(16);
        var b = ((i >> 1) + 0).toString(16);
        colours.push("#" + r + g + b);
    }
    return colours;
}

function hm_on_data(canvas, data){
    var ctx = canvas.getContext("2d");
    var rows = data.rows;
    var scale = canvas.width / (1 << rows[0].key.length);
    ctx.fillStyle = '#f00';
    ctx.strokeStyle = '#0f0';
    var max_value = 0;
    for (var j = 0; j < rows.length; j++){
        var r = rows[j];
        if (r.value > max_value){
            max_value = r.value;
        }
    }
    var colours = make_colour_range();
    var value_scale = (colours.length - 0.01) / max_value;

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
        ctx.fillStyle = colours[parseInt(r.value * value_scale)];
        ctx.fillRect(x * scale, y * scale, scale, scale);
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
