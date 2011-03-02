"""
makeMixture.py - generate data from a mixture of randomly placed Gaussians.
Marcus Frean

Writes a datafile consisting of a matrix in which each row is a
training item. Ground truth is lost in this file: the class is not
written, just the vector.
"""

import sys
from pylab import *
from numpy import *
import numpy.random as rng

#-----------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) == 5:
        D = int(sys.argv[1])
        K = int(sys.argv[2])
        N = int(sys.argv[3])
        out_file = str(sys.argv[4])
        out_stem = out_file.split('.')[0]
        print out_file, out_stem
        print 'dimensionality %d, numClasses %d, numSamples %d' % (D,K,N)
    else:
        sys.exit('usage: python makeMixture.py num_dimensions num_classes num_samples datafilename')
        

    mean = 3 * rng.normal(0.0,1.0,(D,K))   # the centers
    variance = 0.1 + 1.0*rng.random((D,K)) # note the min, max for variances
    prior = 0.2 + 0.8*rng.random((1,K))    # mixing coefficients
    prior = prior / sum(prior)             # normalisation

    covariance = []
    center = []
    for k in range(K):
        center.append(3 * rng.normal(0.0,1.0,(D)))
        c = 0.1 + 1.0*rng.random((D,D))
        c[1,0] = rng.normal(0.0,1.0)
        c[0,1] = c[1,0] #symmetric
        covariance.append(c)

    # generate samples from this mixture of Gaussians
    data = zeros((N,D))
    for i in range(N):
        # choose a component
        j = sum(rng.random() > cumsum(prior))
        # Now choose a data point using that component of the mixture
        x,y = rng.multivariate_normal(center[j],covariance[j],1).T
        data[i,0] = x
        data[i,1] = y
        #for d in range(D):
        #    data[i,d] = transpose(mean)[j,d] + sqrt(variance[d,j]) * rng.normal(0.0,1.0)

    # show the samples as a scatter plot
    scatter(data[:,0], data[:,1], marker='o',s=.5,linewidths=None,alpha=0.5)
    axis('equal')
    draw()
    out_imagename = out_stem+'.png'
    savefig(out_imagename)
    print 'saved image ',out_imagename

    # write the samples to a file
    savetxt(out_file, data, fmt="%12.6G",)
    print 'wrote data file ',out_file
