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

function hm_on_data(canvas, data){
    var ctx = canvas.getContext("2d");
    var rows = data.rows;
    var scale = canvas.width / (1 << rows[0].key.length);
    ctx.fillStyle = '#f00';
    ctx.strokeStyle = '#0f0';
    for (var j = 0; j < rows.length; j++){
        var r = rows[j];
        var n = r.value;
        var coords = r.key;
        var x = 0;
        var y = 0;
        for (var i = 0; i < coords.length; i++){
            /* start from other end */
            var p = parseInt(coords[coords.length - i - 1]);
            x += (p & 1) * (1 << i);
            y += (p >> 1) * (1 << i);
        }
        ctx.fillRect(x * scale, y * scale, n, n);
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
