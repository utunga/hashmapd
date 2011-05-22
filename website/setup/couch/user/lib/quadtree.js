/* lots and lots of mods by hashmapd, eg to remove google  */
/* 1. commented out references to google since that blows up */

/**
 * @name QT.js
 * @author Esa
 * @copyright (c) 2011 Esa I Ojala
 * @fileoverview QT.js is an extension to Google Maps API version 3.
 * It declares a collection of functions for Quadtree indexing
 * http://en.wikipedia.org/wiki/Quadtree
 */


/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/*
Tile numbering scheme follows Z shape.
 ___ ___
| 0 | 1 |
|___|___|
| 2 | 3 |
|___|___|

*/



/**
 *  What is hip? Namespaces are hip. We use 'QT'
 */

var QT = {};

QT.tileSize = 256;


/**
 *  version 0.1 , Quick and ugly first release
 */
QT.VERSION = "0.1";


///**
// *  Mercator projection
// *  can be overwritten like QT.mercator = map.getProjection();
// *  If you have a custom projection or if you just rely more on Google's Mercator projection
// */
//QT.mercator = {};
//QT.mercator.fromLatLngToPoint = function(laLo){
//  var wP = new google.maps.Point();
//  var xx = (laLo.lng() + 180) / 360;
//  wP.x = xx * QT.tileSize;
//  var sinLat = Math.sin(laLo.lat() * Math.PI / 180);
//  var yy = 0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI);
//  wP.y = yy * QT.tileSize;
//  return wP;
//}
//QT.mercator.fromPointToLatLng = function(point){
//  var x = point.x / QT.tileSize - 0.5;
//  var y = point.y / QT.tileSize - 0.5;
//  var lat = 90 - 360 * Math.atan(Math.exp(y * 2 * Math.PI)) / Math.PI;
//  var lng = 360 * x;
//  return new google.maps.LatLng(lat, lng);
//}



/**
 *  encode
 *  @private
 */
QT.encode = function(x, y, z){
  var arr = [];
  for(var i=z; i>0; i--) {
    var pow = 1<<(i-1);
    var cell = 0;
    if ((x&pow) != 0) cell++;
    if ((y&pow) != 0) cell+=2;
    arr.push(cell);
  }
  return arr.join("");
}


/**
 *  Quad Tree key string generator
 *  @param laLo {LatLng}
 *  @return Quadtree key string
 *  also adds '.quad' property to the poor LatLng instance
 */
QT.latLngToQuad = function(laLo){
  var zl = 30;
  var pnt = QT.mercator.fromLatLngToPoint(laLo);
  //var pnt = map.getProjection().fromLatLngToPoint(laLo); // test for comparision
  var tiX = Math.floor(pnt.x * Math.pow(2, zl) / QT.tileSize);
  var tiY = Math.floor(pnt.y * Math.pow(2, zl) / QT.tileSize);
  laLo.quad = QT.encode(tiX, tiY, zl);
  return laLo.quad;
}


///**
// *  a LatLng method .getQuad() attaches '.quad' property and returns that.
// *  LatLng understood as an immutable object
// *  @extends google.maps.LatLng
// */
//google.maps.LatLng.prototype.getQuad = function(){
//  if (this.quad) return this.quad;
//  QT.latLngToQuad(this);
//  return this.quad;
//}



/**
 *  decode a quadtree key string to x, y, z {}
 *  @param quad {String}
 *  @returns {Object} x, y, z
 *  @private I would say. And funny implementation
 */
QT.decode = function(quad){
  var arr = quad.split("");
  var len = arr.length;
  var keyChain = [{x:0, y:0}, {x:1, y:0}, {x:0, y:1}, {x:1, y:1}];
  var xx = yy = 0;
  for (var i=len; i>0; i--){
    var mask = 1 << i;
    xx += keyChain[arr[i-1]].x / mask;
    yy += keyChain[arr[i-1]].y / mask;
  }
  xx *= 1<<len;
  yy *= 1<<len;
  return {x:xx, y:yy, z:len};
}



/**
 *  resolves quadtree key string to LatLng
 *  @param quad {String}
 *  @return {LatLng}
 */
QT.quadToLatLng = function(quad){
  var tile = QT.decode(quad);
  var len = tile.z;
  var wP = {};
  wP.x = QT.tileSize * tile.x / Math.pow(2, len);
  wP.y = QT.tileSize * tile.y / Math.pow(2, len);
  return QT.mercator.fromPointToLatLng(wP);
}


//
///**
// *  @param quad {String}
// *  @returns {LatLngBounds}
// *  A perfect partner for Map.fitBounds()
// */
//
//QT.quadToBounds = function(quad, opt_options){
//  var opts = opt_options || {};
//  var level = quad.length;
//  var part = quad.substring(0, level);
//  var bounds = new google.maps.LatLngBounds(QT.quadToLatLng(part));
//  var SE = QT.nextDoor(quad, 1, 1).substring(0, level);
//  bounds.extend(QT.quadToLatLng(SE));
//  if (opts.visu){                // development aid
//    opts.visu.bounds = bounds;   // option like {visu:{map:map}} visualizes bounds on map
//    bounds.visu = new google.maps.Rectangle(opts.visu);
//  }
//  return bounds;
//}


/**
 *  nextDoor() finds a node in offset to the given node
 *  It doesn't have to be adjacent neighbour. Any offset goes.
 */

QT.nextDoor = function(quad, x_off, y_off){
  var xOff = parseInt(x_off, 10) || 0;
  var yOff = parseInt(y_off, 10) || 0;
  var me = QT.decode(quad);
  var xx = me.x + xOff;
  var yy = me.y + yOff;
  return QT.encode(xx, yy, me.z);
}


/**
 *  A function to shorten a key string.
 *  That is equal to making the square bigger.
 */

QT.clip = function(quad, level){
  var key = quad + "";
  return quad.substring(0, +level);
}


/**
 *  Test and normalize quadtree string. Use this for user input data.
 *  Returns the valid string till the first invalid character
 *  @param quad {string}
 *  @param strict {boolean} If present and true, returns empty string in case of invalid input.
 */

QT.validateQuad = function(quad, strict){
  if (+quad == 0) return quad;   // all zero treatment
  var preZeros = [];             // leading zero treatment
  for (var i=0, len=quad.length; i<len; i++){
    if (quad.charAt(i) != "0") break;
    preZeros.push("0");
  }
  var val = parseInt(quad, 4).toString(4);
  val = preZeros.join("") + val;
  if (strict && quad.length != val.length) return "";
  if (isNaN(+val)) return "";
  return val;
}


/**
 *  utility functions for base-36 conversion and back
 */

QT.base36ToQuad = function(str){
  return parseInt(str, 36).toString(4);
}

QT.quadToBase36 = function(str, opt_prec){
  var temp = str;
  if (opt_prec) temp = temp.substring(0, +opt_prec);
  return parseInt(temp, 4).toString(36);
}



/**
 *  the storage of points and functions to make collision tests
 *  this is the main ToDo part
 */


QT.tree = {};                         // toDo: multiple tree objects
                                      //       include add2tree in .getQuad() ?

QT.add2tree = function(ob){
  var key = ob.getQuad();
  QT.tree[key] = QT.tree[key] || [];  // identical keys handled by array structure
  QT.tree[key].push(ob);
}


QT.isPointInTree = function(obj, lev, callback){
  var quad = obj.getQuad();
  quad = quad.substring(0, +lev);
  var callMe = callback || function(){throw "No callback() in isPointInTree()"};
  var results = [];
  for (var key in QT.tree) {
    if (key.indexOf(quad) == 0) {
      results.push(QT.tree[key]);
    }
  }
  callMe(results);
  return results;
}




