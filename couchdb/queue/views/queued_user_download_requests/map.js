// show only documents that:
//   a) are download requests
//   b) have not yet been completed
//   c) have not been attempted 3 times already
//   	  ... ordered by username
function(doc) {
  if(doc.doc_type == 'download_request' && doc.completed_time == null && (doc.attempts == null || doc.attempts < 3)) {
    emit(doc.username, {'page':doc.page} );
  }
}
