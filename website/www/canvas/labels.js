
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

    /* recentre origin on sx, sy */
    var ox = sx;
    var oy = sy;
    sx = 0;
    sy = 0;
    ex -= ox;
    ey -= oy;
    var len = Math.sqrt(ex * ex + ey * ey);
    var cos = ex / len;
    var sin = ey / len;

    /*go through all the lines and see whether they intersect */
    for (i = 0; i < close_lines.length; i++){
        var line2 = close_lines[i];
        var sx2 = line2[0] - ox;
        var sy2 = line2[1] - oy;
        var ex2 = line2[2] - ox;
        var ey2 = line2[3] - oy;

        /*rotate the system so ex,ey is on the x axis */
        var tmp = sx2 * cos + sy2 * sin;
        sy2 = sy2 * cos - sx2 * sin;
        sx2 = tmp;
        tmp = ex2 * cos + ey2 * sin;
        ey2 = ey2 * cos - ex2 * sin;
        ex2 = tmp;

        if ((sy2 < 0 && ey2 < 0) ||
            (sy2 >= 0 && ey2 >= 0)){
            /* a non-intersect because line2 doesn't cross the x axis*/
            continue;
        }
        /*so it crosses x axis, but where? */
        var xcross = ex2 + (sx2 - ex2) * ey2 / (ey2 - sy2);
        if (xcross < 0 || xcross > len){
            continue;
        }
        return false;
    }
    paint_line(ctx, line[0], line[1], line[2], line[3], '#f00', 1);
    return line;
}

