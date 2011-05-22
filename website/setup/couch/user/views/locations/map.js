 
map = function(doc) {
  
  // NB the comment starting !code below is *not just a comment*
  // it expands into the entirety of 'quadtree.js' because its a couchapp 'directive'
  // see http://wiki.apache.org/couchdb/JavascriptPatternViewCommonJs
  
  // !code lib/quadtree.js
  tmp = QT.tileSize  // just to prove that we can access the library from above
  
  if (doc.doc_type != "twuser") return;
  // FIXME  TO DO- call QT.encode (or something) to emit a 'quadtree key string' instead of just x/y coord
  emit([doc.x_coord, doc.y_coord], 1);
}