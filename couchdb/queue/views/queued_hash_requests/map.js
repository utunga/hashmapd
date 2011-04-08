// show only documents that:
//   a) are hash requests
//   b) have not yet been completed
//   c) have not yet been started
//   	  ... ordered by request time
function(doc) {
  if(doc.doc_type == 'hash_request' && doc.completed_time == null && doc.started_time == null) {
    emit([doc.priority,doc.request_time], {'username':doc.username} );
  }
}
