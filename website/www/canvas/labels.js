
function paint_line(ctx, sx, sy, ex, ey, colour, width){
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
}


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
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.font = size + "px sans-serif";
    var lines = [];
    /* start with the biggest labels */
    points.sort(function(a, b){return b[2] - a[2]});
    for (var i = 0; i < points.length; i++){
        var p = points[i];
        var d = get_pixel_coords(p[0], p[1]);
        if (d.x < 0 || d.x >= $const.width ||
            d.y < 0 || d.y >= $const.height){
            continue;
        }
        var text = p[4];
        var n = p[2];
        var size = Math.log(n) * (1.4 + $state.zoom);
        if (size < 8){
            continue;
        }
        var max_jitter = size / 2 + $state.zoom;
        var line = fit_label(ctx, lines, text, d.x, d.y, size, max_jitter);
        if (line){
            var x = line[5];
            var y = line[6];
            var angle = line[7];
            add_label(ctx, text, x, y, size, "#000"
                      , undefined //"#fff"
                      , angle
            );
            lines.push(line);
        }
    }
    $timestamp("done painting labels");
}

function fit_label(ctx, lines, text, x, y, height, max_jitter){
    var width = ctx.measureText(text).width + height; /*add an M-square*/

    /* Because this is potentially happening a few times (to find a
     * fit) first winnow the list down to those that could possibly
     * intersect.
     */
    var radius = width / 2 + max_jitter + height;
    var close_lines = filter_lines(lines, x, y, radius);
    var angle = 0;
    var jx = 0;
    var jy = 0;
    var rounds = 10;

    for (var j = 0; j < rounds; j++){
        var line = fit_label2(ctx, close_lines, text, x + jx, y + jy, width, height, angle);
        if (line){
            return line;
        }
        var jscale = (rounds + j + j) / (rounds * 3);
        var dj = max_jitter * jscale;
        var da = Math.PI / 2 * jscale;

        jx = Math.random() * dj * 2 - dj;
        jy = Math.random() * dj * 2 - dj;
        angle = Math.random() * da * 2 - da;
    }
    return false;
}

function fit_label2(ctx, close_lines, text, x, y, width, height, angle){
    var dx = Math.cos(angle);
    var dy = Math.sin(angle);
    var sx = x - dx * width / 2;
    var sy = y - dy * width / 2;
    var ex = sx + dx * width;
    var ey = sy + dy * width;
    var line = [sx, sy, ex, ey, height * 0.75, x, y, angle];

    /* recentre origin on sx, sy */
    var ox = sx;
    var oy = sy;
    sx = 0;
    sy = 0;
    ex -= ox;
    ey -= oy;
    var cos = ex / width;
    var sin = ey / width;

    /*go through all the lines and see whether they intersect */
    for (var i = 0; i < close_lines.length; i++){
        var line2 = close_lines[i];
        var sx2 = line2[0] - ox;
        var sy2 = line2[1] - oy;
        var ex2 = line2[2] - ox;
        var ey2 = line2[3] - oy;
        var padding = line2[4];
        /*rotate the system so ex,ey is on the x axis */
        var tmp = sx2 * cos + sy2 * sin;
        sy2 = sy2 * cos - sx2 * sin;
        sx2 = tmp;
        tmp = ex2 * cos + ey2 * sin;
        ey2 = ey2 * cos - ex2 * sin;
        ex2 = tmp;
        if ((sy2 < -padding && ey2 < -padding) ||
            (sy2 >= padding && ey2 >= padding)){
            /* a non-intersect because line2 doesn't cross the x axis*/
            continue;
        }
        /*so it crosses x axis, but where?
         * first check for vertical */
        var xcross_low, xcross_high;
        if (ey2 == sy2){
            xcross_low = ey2 - padding;
            xcross_high = ey2 + padding;
        }
        else{
            /* padding is skewed by angle &
               division by (ey2 - sy2) is safe. */
            var dx2 = ex2 - sx2;
            var dy2 = ey2 - sy2;
            var xcross = ex2 - dx2 * ey2 / dy2;
            var p = Math.abs(padding * dx2 / dy2);
            xcross_high = xcross + p;
            xcross_low = xcross - p;
        }
        if (xcross_high < 0 || xcross_low > width){
            continue;
        }
        return false;
    }
    //paint_line(ctx, line[0], line[1], line[2], line[3], '#f00', 1);
    return line;
}


function filter_lines(lines, x, y, radius){
    var close_lines = [];
    var lx = x - radius;
    var hx = x + radius;
    var ly = y - radius;
    var hy = y + radius;

    for (var i = 0; i < lines.length; i++){
        var line = lines[i];
        if ((line[0] > hx && line[2] > hx) ||
            (line[0] < lx && line[2] < lx) ||
            (line[1] > hy && line[3] > hy) ||
            (line[1] < ly && line[3] < ly)){
            continue;
        }
        close_lines.push(line);
    }
    //log("found", close_lines.length, "close lines out of", lines.length);
    return close_lines;
}
