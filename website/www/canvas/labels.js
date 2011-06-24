
function paint_labels(){
    $timestamp("painting labels");
    var points = $page.labels.points;
    var canvas = overlay_canvas("labels", undefined, true);
    var ctx = canvas.getContext("2d");
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        //log(p);
        var d = get_pixel_coords(p[0], p[1]);
        if (d.x < 0 || d.x >= $const.width ||
            d.y < 0 || d.y >= $const.height){
            continue;
        }
        var text = p[4];
        var n = p[2];
        var jitter = $const.COORD_MAX >> (p[3] + 7);
        var size = Math.log(n) * (1.3 + $state.zoom);
        var jx = Math.random() * jitter * 2 - jitter;
        var ja = Math.random() * Math.PI - Math.PI / 2;
        var jy = Math.random() * jitter * 2 - jitter;
        add_label(ctx, text, d.x + jx, d.y + jy, size, "#000"
                  , undefined //, "#000"
                  , ja
                 );
    }
    $timestamp("done painting labels");
}

function add_label(ctx, text, x, y, size, colour, shadow, angle){
    //log.apply(undefined, arguments);
    if (colour === undefined){
        colour = "#000";
    }
    if (angle){
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);
        x = 0; y = 0;
    }
    if (shadow && size > 8){
        ctx.shadowColor = shadow;
        ctx.shadowBlur = size * 0.125;
        ctx.shadowOffsetX = 0.5 + size * 0.125;
        ctx.shadowOffsetY = 0.5 + size * 0.125;
    }
    //ctx.globalAlpha = "0.6";
    ctx.textAlign = "center";
    ctx.font = size + "px sans-serif";
    ctx.fillStyle = colour;
    ctx.fillText(text, x, y);

    if (angle){
        ctx.restore();
    }
}

function paint_labels_cleverly(){
    $timestamp("painting labels");
    var points = $page.labels.points;
    var canvas = overlay_canvas("labels", undefined, true);
    var ctx = canvas.getContext("2d");
    var lines = [];
    /* start with the biggest labels */
    points.sort(function(a, b){return b[2] - a[2]});
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        //log(p);
        var d = get_pixel_coords(p[0], p[1]);
        if (d.x < 0 || d.x >= $const.width ||
            d.y < 0 || d.y >= $const.height){
            continue;
        }
        var text = p[4];
        var n = p[2];
        var size = Math.log(n) * (1.3 + $state.zoom);
        if (size < 8){
            continue;
        }
        var jitter = $const.COORD_MAX >> (p[3] + 7);
        for (var j = 0; j < 10; j++){
            //var angle = Math.PI * j / 20;
            var angle = Math.PI / 2;
            var jx = Math.random() * jitter * 2 - jitter;
            var ja = Math.random() * angle * 2 - angle;
            var jy = Math.random() * jitter * 2 - jitter;
            var line = fit_label(ctx, lines, text, d.x, d.y, size, "#000", ja, jitter);
            if (line){
                //log("adding " + line + " on iteration", j);
                add_label(ctx, text, d.x + jx, d.y + jy, size, "#000"
                          , undefined //, "#000"
                          , ja
                         );
                lines.push(line);
                break; /*continue main loop */
            }
        }
    }
    $timestamp("done painting labels");
}

function paint_line(ctx, sx, sy, ex, ey, colour, width){
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
}

function fit_label(ctx, lines, text, x, y, size, colour, angle, max_jitter){
    /*XXX could get smaller rectangle */
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.font = size + "px sans-serif";
    var w = ctx.canvas.width;
    var h = ctx.canvas.height;
    var i;
    //var pix = ctx.getImageData(0, 0, w, h).data;
    var dx = Math.cos(angle);
    var dy = Math.sin(angle);
    var width =  ctx.measureText(text).width + size; /*add an M-sqaure*/
    var sx = x - dx * width / 2;
    var sy = y - dy * width / 2;
    var ex = sx + dx * width;
    var ey = sy + dy * width;
    var line = [sx, sy, ex, ey, size / 2];

    /*Because this is potentially happening a few times (to find a fit)
     winnow the list down to those that could possibly intersect.
     */
    var close_lines = [];
    var lx = x - width / 2 - max_jitter;
    var hx = lx + width + max_jitter * 2;
    var ly = y - width / 2 - max_jitter;
    var hy = ly + width + max_jitter * 2;

    for (var i = 0; i < lines.length; i++){
        var line2 = lines[i];
        var hx2 = Math.max(line2[0], line2[2]);
        var hy2 = Math.max(line2[1], line2[3]);
        var lx2 = Math.min(line2[0], line2[2]);
        var ly2 = Math.min(line2[1], line2[3]);
        if (lx2 > hx ||
            hx2 < lx ||
            ly2 > hy ||
            hy2 < ly){
            continue;
        }
        close_lines.push(line2);
    }
    log("found", close_lines.length, "close lines out of", lines.length);
    /*go through all the lines and see whether they intersect */
    for (i = 0; i < close_lines.length; i++){
        var line2 = close_lines[i];
        var sx2 = line2[0];
        var sy2 = line2[1];
        var ex2 = line2[2];
        var ey2 = line2[3];
        if (intersect(sx, sy, ex, ey, sx2, sy2, ex2, ey2)){
            return false;
        }
    }
    paint_line(ctx, sx, sy, ex, ey, '#f00', 1);
    return line;
}


function intersect(sx1, sy1, ex1, ey1, sx2, sy2, ex2, ey2){
    var  cos, sin, newX, ABpos;
    //  (1) Translate the system so that point A is on the origin.
    /*re-zero on sx1, sy1 */
    ex1 -= sx1;
    ey1 -= sy1;
    ex2 -= sx1;
    ey2 -= sy1;
    sx2 -= sx1;
    sy2 -= sy1;
    sx1 = 0;
    sy1 = 0;

    //  Discover the length of segment A-B.
    var len = Math.sqrt(ex1 * ex1 + ey1 * ey1);

    //  (2) Rotate the system so that point B is on the positive X axis.
    cos = ex1 / len;
    sin = ey1 / len;
    var tmp = sx2 * cos + sy2 * sin;
    sy2 = sy2 * cos - sx2 * sin;
    sx2 = tmp;
    tmp = ex2 * cos + ey2 * sin;
    ey2 = ey2 * cos - ex2 * sin;
    ex2 = tmp;

    //  Fail if segment C-D doesn't cross line A-B.
    if ((sy2 < 0 && ey2 < 0) ||
        (sy2 >= 0 && ey2 >= 0)){
        return false;
    }

    //  (3) Discover the position of the intersection point along line A-B.
    ABpos = ex2 + (sx2 - ex2) * ey2 / (ey2 - sy2);

    //  Fail if segment C-D crosses line A-B outside of segment A-B.
    if (ABpos < 0. || ABpos > len){
        return false;
    }
    return true;
}


function intersect2(sx1, sy1, ex1, ey1, sx2, sy2, ex2, ey2) {
    /*re-zero on sx1, sy1 */
    ex1 -= sx1;
    ey1 -= sy1;
    ex2 -= sx1;
    ey2 -= sy1;
    sx2 -= sx1;
    sy2 -= sy1;
    sx1 = 0;
    sy1 = 0;
    var dx2 = ex2 - sx2;
    var dy2 = ey2 - sy2;

    var slope = ex1 * dx2 - ey1 * dy2;
    if (Math.abs(slope) < 0.0001){
        /*slopes are identical
         *XXX should check for overlap */
        return false;
    }
    var x = (ex2 * ey1 - ey2 * ex2) / slope;
    var y = (sx2 * ey1 - sy2 * ex2) / slope;
    log(x, y);
    if(x < 0 || y < 0 || x > 1 || y > 1){
        return false;
    }
    return true;
}


function intersect3(){
    /* so here they are in the little quadrant defined by this label.
     They intersect if line2 crosses the diagonal defined by the line.
     That is, its sx, sy is on the other side from its ex, ey.

     If the rectangle is scaled to square, and ey > sy, and the
     points are normalised to put sx,sy at 0,0, then the known
     line is on x == y.  The other line must intersect if
     sx / sy > 1 and ex /ey < 1 or vice versa
     and not if they are both on the same side of 1.
     To get rid of zero division:
     sx2r /sy2r > 1 ==> sx2r / sy2r > exr / eyr
     ==> sx2r * eyr > exr * sy2r

     and of course there is no need to scale to a square any more.

     but if ey < sy, it is different. Now what matters is whether
     they're on the same side of y = 1 - x.




     if ey == sy, intersection is if sy2r and ey2r are on opposite sides.


     */
        /* normalise everybody into zero based square. our line is now 0,0 <--> exr, exr*/
    var sx2r = line2[0] - sx;
    var sy2r = line2[1] - sy;
    var ex2r = (line2[2] - sx);
    var ey2r = (line2[3] - sy);

    /*
     var shigh = (sx2r * eyr > exr * sy2r);
     var ehigh = (ex2r * eyr > exr * ey2r);
     if (shigh != ehigh)
     {
     // a crossing!
     log(line2, "intesects with", line);
     paint_line(ctx, sx, sy, ex, ey, '#f00', 1);
     return false;
     }
     paint_line(ctx, sx, sy, ex, ey, '#afa', 3);
     continue;
     */



    if(1){
    /*rotate so that ex2, ey2 is on the x axis. */
        var sx2rr = sx2r * cos + sy2r * sin;
        var sy2rr = sy2r * cos + sx2r * sin;
        var ex2rr = ex2r * cos + ey2r * sin;
        var ey2rr = ey2r * cos + ex2r * sin;

        if ((sy2rr < 0 && ey2rr < 0) || (sx2rr < 0 && ex2rr < 0)){
            /*no intersection !*/
            paint_line(ctx, sx, sy, ex, ey, '#afa', 3);
            continue;
        }
        /*distance along line of intersection*/
        var d = ex2r + (sx2r - ex2r) * ex2r / (ey2r - sy2r);
        if (d < 0 || d > len){
            /*no intersection */
            //paint_line(ctx, sx, sy, ex, ey, '#aaf', 3);
            continue;
        }
        return false;
        }
}