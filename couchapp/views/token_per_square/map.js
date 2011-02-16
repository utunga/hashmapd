function(doc) {
	x = Math.round(doc.x_coord);
	y = Math.round(doc.y_coord);
	key = [x,y, doc.token, doc.overall_count];
	value = doc.word_prob
  	emit(key, value);
}