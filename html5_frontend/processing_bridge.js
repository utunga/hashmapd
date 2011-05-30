
/* global variables for processing to find. */
var hm_points = [];
var hm_points_loaded = false;

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
    for (i = 0; i < rows.length; i++){
        var r = rows[i];
        var coords = r.key;
        var x = 0;
        var y = 0;
        for (j = 0; j < coords.length; j++){
            /* start from other end */
            var p = coords[coords.length - j - 1];
            x <<= 1;
            y <<= 1;
            x += (p & 1);
            y += (p >> 1);
        }
        hm_points.push([x, y, r.value]);
    }
    /* A flag, just in case processing tries loading hm_points before
       the preceding loop is done. */
    hm_points_loaded = true;
}

