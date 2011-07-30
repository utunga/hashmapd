function(doc) { 
  doc_type = (doc.doc_type !== "undefined") ? doc.doc_type : "unknown";
  if (doc_type == "raw_tweet") {
    emit([doc.username, 1, doc._id], doc);
  }
  if (doc_type == "user") {
    emit([doc.screen_name, 0, doc._id], doc);
  }

}
