// this view takes tweet data that was  generated in csharp, and creates an entry for each
//    user (should be run with group = True)
function(doc) {
  emit([doc.screen_name,{screen_name:doc.screen_name,doc_type:"user",hash:null,coords:null}],null)
}