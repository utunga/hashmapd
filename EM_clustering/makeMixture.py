"""
makeMixture.py - generate data from a mixture of randomly placed Gaussians.
Marcus Frean

Writes a datafile consisting of a matrix in which each row is a
training item. Ground truth is lost in this file: the class is not
written, just the vector.
"""

import sys, math
import pylab as pl
import numpy as np
import numpy.random as rng

#-----------------------------------------------------------------------------

def makeRandom2DCovMatrix():
    # first make some Gaussian data in D dimensions
    dumData = rng.normal(0.0,0.5+2.0*rng.random(2),(200,2))
    # rotate it a bit (easy enough when it's just 2D)
    angle = math.pi * rng.random()
    rotMatrix = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])
    dumData = np.dot(dumData,rotMatrix)
    # find the covariance matrix!
    v00 = np.mean(dumData[:,0]*dumData[:,0])
    v01 = np.mean(dumData[:,0]*dumData[:,1])
    v10 = np.mean(dumData[:,1]*dumData[:,0])
    v11 = np.mean(dumData[:,1]*dumData[:,1])
    covarMatrix = np.array([[v00,v01],[v10,v11]])
    return covarMatrix


if __name__ == '__main__':

    if len(sys.argv) == 5:
        D = int(sys.argv[1])
        if (D != 2):
            sys.exit('only sure this works for 2D at present - sorry!')
        K = int(sys.argv[2])
        N = int(sys.argv[3])
        out_file = str(sys.argv[4])
        out_stem = out_file.split('.')[0]
        print out_file, out_stem
        print 'dimensionality %d, numClasses %d, numSamples %d' % (D,K,N)
    else:
        sys.exit('usage: python makeMixture.py num_dimensions num_classes num_samples datafilename')
        

    mean = 5 * rng.normal(0.0,1.0,(D,K))   # the centers
    variance = 0.1 + 1.0*rng.random((D,K)) # note the min, max for variances
    prior = 0.2 + 0.8*rng.random((1,K))    # mixing coefficients
    prior = prior / np.sum(prior)             # normalisation

    covariance = []
    center = []
    for k in range(K):
        newMean = 5.0 * rng.normal(0.0,1.0,(D))
        center.append(newMean)
        c = makeRandom2DCovMatrix()
        covariance.append(c)

    # generate samples from this mixture of Gaussians
    data = np.zeros((N,D))
    for i in range(N):
        # choose a component
        j = np.sum(rng.random() > np.cumsum(prior))
        # Now choose a data point using that component of the mixture
        x,y = rng.multivariate_normal(center[j],covariance[j],1).T
        data[i,0] = x
        data[i,1] = y
        #for d in range(D):
        #    data[i,d] = transpose(mean)[j,d] + sqrt(variance[d,j]) * rng.normal(0.0,1.0)

    # show the samples as a scatter plot
    pl.scatter(data[:,0], data[:,1], marker='o',s=.5,linewidths=None,alpha=0.5)
    pl.axis('equal')
    pl.draw()
    out_imagename = out_stem+'.png'
    pl.savefig(out_imagename)
    print 'saved image ',out_imagename

    # write the samples to a file
    np.savetxt(out_file, data, fmt="%12.6G",)
    print 'wrote data file ',out_file
