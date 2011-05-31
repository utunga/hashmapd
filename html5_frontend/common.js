var DATA_URL = 'locations.json';
var HTTP_OK = 200;
var XML_HTTP_READY = 4;

/** hm_draw_map is the main entrance point.
 *
 * Nothing happens until the json is loaded, then the hm_on_data
 * function is called with the canvas reference and JSON
 * data. hm_on_data is differently defined for processing and bare
 * canvas implementations..
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
