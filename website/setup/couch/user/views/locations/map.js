
map = function(doc) {
  if (doc.doc_type != "twuser") return;
    /* Scales:
     *  7:  128 x 128
     *  8:  256 x 256
     *  9:  512 x 512
     * 10:  1024 x 1024
     * 11:  2048 x 2048
     * 12:  4096 x 4096
     * */

  MAX_ZOOM = 9;

  var quadKey = [];
  var x = doc.x_coord;
  var y = doc.y_coord;
  for(var i = MAX_ZOOM -1; i >= 0; i--) {
    quadKey[i] = ((x & 1) + (y & 1) * 2);
    x >>= 1;
    y >>= 1;
  }
  emit(quadKey, 1);
}
