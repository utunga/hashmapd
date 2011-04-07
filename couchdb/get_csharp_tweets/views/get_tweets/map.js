// this view takes tweet data that was  generated in csharp, and mimicks the structure of tweets stored
//    in the hashmapd database
function(doc) {
  emit(doc.twitter_id.toString(),{username:doc.screen_name,id:doc.twitter_id,id_str:doc.twitter_id.toString(),provider_namespace:"twitter",text:doc.text,entities:doc.entities,doc_type:"raw_tweet",from_csharp:"true"});
}