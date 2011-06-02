function(doc) {
    /* NB the comment starting !code below is *not just a comment* it
       expands into the entirety of 'hashmapd_quadtree.js' because it
       is a couchapp 'directive' see
       http://wiki.apache.org/couchdb/JavascriptPatternViewCommonJs
    */
    // !code lib/hashmapd_quadtree.js

    /* The value is a [token, wordcount] pair, can be reduced with
     * nearby tokens by summing the word counts.
     */
    if (doc.doc_type == "token_n_coords"){
        var coords = convert_coords(doc.x_coord, doc.y_coord);
        if (coords !== undefined){
            emit(coords, [doc.token, doc.user_wordcount]);
        }
    }
}
