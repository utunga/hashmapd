
function(doc) {}

// removes duplicate keys, and retains the first value
/*
function(keys, values) {
  var dups = new Array();
  var x = 0;
  for (i = 0; i < values.length; i++) {
    if (!dups[keys[i]]) {
	  dups[keys[i]] = 1;
	  // emit(keys[i],values[i]);
	  x+=1;
	}
  }
  emit(x,{});
}
*/