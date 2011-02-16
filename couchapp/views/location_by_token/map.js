function(doc) {
  emit([doc.token, doc.x_coord, doc.y_coord], doc.count);
}