map = function(doc) {
    /* NB the comment starting !code below is *not just a comment*
       it expands into the entirety of 'quadtree.js' because its a couchapp 'directive'
       see http://wiki.apache.org/couchdb/JavascriptPatternViewCommonJs
    */
    // !code lib/hashmapd_quadtree.js

    if (doc.doc_type == "twuser"){
        var coords = convert_coords(doc.x_coord, doc.y_coord);
        if (coords !== undefined){
            emit(coords, 1);
        }
    }
}
