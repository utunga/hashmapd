// show only documents that:
//   a) are hash requests
//   b) have not yet been completed
//   c) have been started (are underway)
//   	  ... ordered by start time
function(doc) {
  if(doc.doc_type == 'hash_request' && doc.completed_time == null && doc.started_time != null) {
    emit([doc.priority,doc.started_time], {'username':doc.username} );
  }
}
