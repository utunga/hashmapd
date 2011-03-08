import numpy, time, cPickle, gzip, sys, os

import math
import theano
import theano.tensor as T

if __name__ == '__main__':
    total = 1;
    docSize = 50;
    mean = 0.1;
    
    x = theano.shared(value = numpy.array([0,0.2,0.5,0.1,0.05,0.02,0.1,0.02,0.01,0]), name = 'x')
    y = theano.shared(value = float(1000), name = 'y');
    
    # approximate poisson distribution using a binomial distribution
    rng = theano.tensor.shared_randomstreams.RandomStreams(numpy.random.RandomState().randint(2**30));
    n = T.max(x,axis=range(1))*y;
    p = rng.binomial(x.shape,n,x*docSize/n);
    poisson = theano.function([],p);
    
    # test
    result = poisson();
    
    raw_counts = {}
    
    # p(k|mean) = mean^k*exp(-mean)/k!
    
    print "len: "+str(len(result));
    
    for i in range(0,10) :
        raw_counts[i] = result[i];
        
    for k,v in raw_counts.items() :
        print str(k)+" = "+str(v);

