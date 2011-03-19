function(doc) {
  emit([doc.token, doc.screen_name], doc.word_prob);
}