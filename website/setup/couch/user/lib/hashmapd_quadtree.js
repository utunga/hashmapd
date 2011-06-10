/* Translate, scale, truncate, and quad-tree-ise coordinates,
 * discarding any ridiculous outliers.
 *
 * It is important for the coordinates to all be POSITIVE integers and
 * for their upper bound to be known POWER OF 2.  Otherwise you get
 * rubbish.  That is the reason for all this scaling, translation, and
 * winnowing.
 */

/* NOTE: this code gets included *within a function body*, but that
 * doesn't particularly limit what you can do.
 */

/*should the map be turned upside down?  (If so, make sure the various
bounds and translations take this into account, as the default case
does (it is symmetrical).  */
var Y_FLIP = true;

/* {X,Y}_TRANSLATE and the various BOUNDs don't need to be powers of
 * 2, or even integers, or the same as each other.  It just happens
 * that all of these things are true for now.
 */
var X_TRANSLATE = 128;
var Y_TRANSLATE = 128;

/*bounds checking is done BEFORE translation.  The lower bounds can't
 * be smaller than the coreesponding translation, or negative numbers
 * might sneak through.
 The bounds themselves are excluded.
 */
var X_BOUND_LOW = -X_TRANSLATE;
var X_BOUND_HIGH = X_TRANSLATE;

var Y_BOUND_LOW = -Y_TRANSLATE;
var Y_BOUND_HIGH = Y_TRANSLATE;

/*what is the ultimate scale (in bits) / number of coordinates
 * 12 ->  4k square
 * 13 ->  8k   "
 * 14 -> 16k   "
 * 15 -> 32k   "
 * 16 -> 64k   "
 *
 * group_level can't go deeper than 12, so you can get 1-12, or all
 * coordinates (using group=exact).
 */
var MAX_QUAD_COORDS = 15;
var X_SCALE = (1 << MAX_QUAD_COORDS) / (X_BOUND_HIGH - X_BOUND_LOW);
var Y_SCALE = (1 << MAX_QUAD_COORDS) / (Y_BOUND_HIGH - Y_BOUND_LOW);

function convert_coords(x, y){
    if (Y_FLIP) {
        y = -y;
    }
    if (x <= X_BOUND_LOW ||
        x >= X_BOUND_HIGH ||
        y <= Y_BOUND_LOW ||
        y >= Y_BOUND_HIGH) {
        return undefined;
    }
    x += X_TRANSLATE;
    y += Y_TRANSLATE;
    x = parseInt(x * X_SCALE);
    y = parseInt(y * Y_SCALE);
    var quadkey = [];
    for (var i = MAX_QUAD_COORDS; i > 0; i--) {
        quadkey[i - 1] = (2 * (y & 1) + (x & 1));
        x >>= 1;
        y >>= 1;
    }
    return quadkey;
}
