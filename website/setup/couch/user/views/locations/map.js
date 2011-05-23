 
map = function(doc) {
  if (doc.doc_type != "twuser") return;
  
  // NB the comment starting !code below is *not just a comment*
  // it expands into the entirety of 'quadtree.js' because its a couchapp 'directive'
  // see http://wiki.apache.org/couchdb/JavascriptPatternViewCommonJs
  
  // !code lib/quadtree.js
  tmp = QT.tileSize  // just to prove that we can access the library from above
   // weird things happen if you set this too high - the number starts
   // repeating *not in the last place, as you might expect, but in the first place*
   // by trial and error this seems to work for the float coords between -100 and +100 we have currently
  MAX_ZOOM = 7;
  quadKey = QT.encode(doc.x_coord, doc.y_coord, MAX_ZOOM);
  emit(quadKey, 1);
}