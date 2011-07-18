function(doc) {
    /* super quick extrqct of just xbox data
     */
    if (doc.doc_type == "token_n_coords"){
        if (doc.token=='Xbox')
            emit(null, doc);
    
    }
}
