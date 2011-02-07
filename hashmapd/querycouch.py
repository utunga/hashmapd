
import sys
import getopt
import os
import csv
import cPickle
import gzip
import theano
import time, PIL.Image
import couchdb

from csv_unicode_helpers import UnicodeWriter
from struct import *
from numpy import *
from couchdb.mapping import Document, LongField, DateField, FloatField, TextField, IntegerField, BooleanField
from couchdb import Server

class QueryCouch(object):
        
        def __init__(self, cfg):
                couch = Server(cfg.couchdb.server_url)
                self.db = couch[cfg.couchdb.database]
            
        def tokens_for_square(self, coord_x, coord_y):
            view = self.db.view('tokens_per_square/tokens_per_square', reduce=False)
                
            results = []
            rows = view[[coord_x,coord_y]:[coord_x,coord_y+1]]
            for row in rows:
                results.append((row.value,row.key[2]))
                
            results.sort()
            results.reverse()
            return results
        
        def top_token_for_square(self, x_coord, y_coord):
            results = self.tokens_for_square(x_coord, y_coord)
            if (len(results)>0):
                #print (x_coord, y_coord, results[0][1])
                return (x_coord,y_coord,results[0][1])
            return (x_coord,y_coord)
            
        def squares_for_token(self, token):
            view = self.db.view('square_per_token/square_per_token', reduce=False)
                
            results = []
            rows = view[[token]:[token+' ']]
            for row in rows:
                results.append((row.value,row.key))
                
            results.sort()
            results.reverse()
            return results
        
        def top_square_for_token(self, token):
            results = self.squares_for_token(token)
            if (len(results)>0):
                token=results[0][1][0]
                x_coord=results[0][1][1]
                y_coord=results[0][1][2]
                return (x_coord,y_coord,token)
          
        def all_tokens(self, topN=None):
            
            # we have to query entire lot of tokens even if we only want topN as the sort has to be on Python side
            # (because CouchDB doesn't support sort by value)
            view = self.db.view('square_per_token/square_per_token', reduce=True, group_level=1)
           
            results = []
            for row in view:
                results.append((row.value,row.key))
                
            results.sort()
            results.reverse()
            if (topN==None):
                    return results
            else:
                    return results[0:topN]
                    
        def locations_for_token(self, token):
                
            view = self.db.view('location_by_token/location_by_token', reduce=False)
                
            results = []
            rows = view[[token]:[token+' ']]
            for row in rows:
                    results.append((row.key[1], row.key[2]))
            return results