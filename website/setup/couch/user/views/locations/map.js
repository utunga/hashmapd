
map = function(doc) {
    if (doc.doc_type != "twuser") return;
    emit([doc.x_coord, doc.y_coord], 1);
}
