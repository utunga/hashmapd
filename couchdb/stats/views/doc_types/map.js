function(doc) { 
  doc_type = (doc.doc_type !== "undefined") ? doc.doc_type : "unknown";
  emit(doc_type, 1);
  emit("total",1);
}
