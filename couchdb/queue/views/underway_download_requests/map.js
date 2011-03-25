// show only documents that:
//   a) are download requests
//   b) have not yet been completed
//   c) have not been started within the last FIVE minutes
//   	  ... ordered by request time
function(doc) {
  if(doc.type == 'download_request' && doc.completed_time == null && doc.started_time != null) {
    emit(doc.started_time, {'_id':doc._id,'username':doc.screen_name, 'page':doc.page} );
  }
}
