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
  
  var text = doc.text;
  var screen_name = doc.screen_name;

  var body = stripNonWords(text).toLowerCase();
  var terms = [];
  var words = body.split(/\s+/);
 
  var i = words.length;
  while (i--) {
    var word = words[i];
    if(word.length> 2) {
      if(!terms.contains(word)) {
        terms.push(word);
          var weight = words.count(word);
         
          var type = "word";
          if (word.substring(0,1)=='#') {
            type = 'tag'; 
          } else if (word.substring(0,1)=='@') {
            type = 'mention'
          }
          emit([screen_name, word], weight);
        
      }
    }
  }
}
