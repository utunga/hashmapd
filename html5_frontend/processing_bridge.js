
/* global variables for processing to find. */
var hm_data = {
    points: [],
    points_loaded: false,
    max_value: 0,
    scale: 1,
    PADDING: 16
}

/* for debugging processing. Push things on here for Chromium debugger
 * to see. */

var hm_debug = [];

function hm_on_data(canvas, data){
    /* The natural way to do this would be to call something like this:

     var pc = Processing.getInstanceById(canvas.id);
     pc.addPoint(...);

     BUT the Processing instances don't necessarily exist at this
     point.  If the JSON is quick it wins the race.

     So instead, save to a temporary object, then let processing ask
     for it when it is ready.
     */
    var i, j;
    var rows = data.rows;
    var max_value = 0;
    for (i = 0; i < rows.length; i++){
        var r = rows[i];
        var coords = r.key;
        var x = 0;
        var y = 0;
        for (j = 0; j < coords.length; j++){
            var p = coords[j];
            x = (x << 1) | (p & 1);
            y = (y << 1) | (p >> 1);
        }
        hm_data.points.push([x, y, r.value]);
        if (r.value > max_value){
            max_value = r.value;
        }
    }
    /* A flag, just in case processing tries loading hm_data.points before
       the preceding loop is done. */
    hm_data.points_loaded = true;
    hm_data.max_value = max_value;
    hm_data.scale = (canvas.width - 2 * hm_data.PADDING) / (1 << rows[0].key.length);
}
