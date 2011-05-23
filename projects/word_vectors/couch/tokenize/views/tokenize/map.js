Array.prototype.contains = function(obj) {
  var i = this.length;
  while (i--) {
    if (this[i] === obj) {
      return true;
    }
  }
  return false;
}
 
 
Array.prototype.count = function(obj) {
  var count = 0;
  var i = this.length;
  while (i--) {
    if (this[i] === obj) {
      count++;
    }
  }
  return count;
}
 
 
function stripNonWords(w){
  return w.replace(/[^a-z\@\#\_1-9A-Z]+/ig," ");
}

 
map = function(doc) { 
  if (doc.doc_type != "raw_tweet") return;
  
  var body = stripNonWords(doc.text).toLowerCase();
  var terms = [];
  var words = body.split(/\s+/);
 
  var i = words.length;
  while (i--) {
    var word = words[i];
    if(word.length> 2) {
      if(!terms.contains(word)){
        terms.push(word);
        var weight = words.count(word);
        
          emit([doc.username, word], weight);
        
      }
    }
  }
}