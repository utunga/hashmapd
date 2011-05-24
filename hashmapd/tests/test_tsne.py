
import numpy, time, cPickle, gzip, sys, os

import math
import theano
import theano.tensor as T

if __name__ == '__main__':
    
    # checking this in more as documentation than as a working test
    # pretty sure this wouldn't even begin to work as it is
    
    #when running 'offline':
    codes = smh.run()
    tsne = TSNE(perplexity=10)
    tsne.load_codes(codes)
    tsne.fit(iterations=1000)
    tsne.save_codes_to_file()
    
    #when running 'online'
    user_hist = couch_wrapper.get_user_data(username)
    tsne = TSNE(perplexity=10)
    tsne.load_codes_from_file()
    code = smh.get_code_for_user_hist(user_hist)
    coord = tsne.get_2d(code, iterations=100)
    #send the coord back to user
    
    #every couple of weeks
    tsne = TSNE(perplexity=10)
    tsne.load_codes_from_file()
    tsne.fit(iterations=100)
    tsne.save_codes_to_file()
    