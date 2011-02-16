function(doc) {
	x = Math.round(doc.x_coord);
	y = Math.round(doc.y_coord);
	key = [doc.token, x,y];
	value = doc.word_prob
  	emit(key, value);
}