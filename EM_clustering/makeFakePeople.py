"""
This is similar to makeMixture.py - generate data from a mixture of
randomly placed 2D Gaussians.

But in this case it also makes up a random histogram of words,
associated with each component of the mixture.

After generating some 2D points from the MoG model, we can go through
them point-by-point figuring out a 'personal' histogram for each of
them, by just taking the component histograms and weighting them
differently based on the responsibilities that EM would assign the
point. This ensures a smooth interpolation of histograms across the
map.

I then store this as a List of Person objects.

Oh and make some pictures as a reality check.

Marcus
"""

import sys, pickle
import pylab as pl
import numpy as np
import numpy.random as rng
from makeMixture import makeRandom2DCovMatrix

#--------------------------------------------------------------------------
class Person:
    """ a person is a point in 2d space, and a histogram of word usage """
    x,y = 0.0,0.0
    histo = []
    
    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.histo = h

    def __str__(self):
        print("x %.2f, y %.2f, histo: %s" % (self.x,self.y,self.histo))
#--------------------------------------------------------------------------

def GaussDensity(d,mu,v):
    """
    returns the prob of data d (matrix with items as rows), under the
    Gaussian given by mu (vector) and var (matrix)
    """
    if (np.linalg.det(v) < 0.0):
        sys.exit('determinant is negative! ')
    normalisation = 2*np.pi * np.sqrt(np.linalg.det(v))
    xx = d-mu
    yy = np.dot(np.linalg.inv(v), xx.transpose()).transpose()
    s1 = xx[0] * yy[0]
    s2 = xx[1] * yy[1]
    u = -0.5*(s1+s2)
    p = np.exp(u)/normalisation 
    return p

#-----------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) == 4:
        K = int(sys.argv[1])
        N = int(sys.argv[2])
        out_file = str(sys.argv[3])
        out_stem = out_file.split('.')[0]
        print out_file, out_stem
        print 'numClasses %d, numSamples %d' % (K,N)
    else:
        sys.exit('usage: python makePeople.py num_classes num_samples datafilename')
        

    D = 2 # data will be 2 dimensional
    mean = 3 * rng.normal(0.0,1.0,(D,K))   # the centers
    variance = 0.1 + 1.0*rng.random((D,K)) # note the min, max for variances
    mix_coeff = 0.2 + 0.8*rng.random((K))# mixing coefficients
    mix_coeff = mix_coeff / np.sum(mix_coeff) # normalisation

    covarianceMats = []
    centers = []
    vocabulary = ['cleansing','waterboard','heaven','xbox']
    histograms = []
    for k in range(K):
        centers.append(5.0* rng.normal(0.0,1.0,(D)))
        c = makeRandom2DCovMatrix()
        covarianceMats.append(c)
        histograms.append(np.power(rng.random((len(vocabulary))),2.0))

    print '================================================'
    # generate samples from this mixture of Gaussians, and give each a
    # histogram, and store these in a Person object.
    people = []
    for i in range(N):
        # choose a component
        j = np.sum(rng.random() > np.cumsum(mix_coeff))
        # Now choose a data point using that component of the mixture
        x,y = rng.multivariate_normal(centers[j],covarianceMats[j],1).T
        point = np.array([x,y])
        point.shape = (2,)

        # now make up a histogram to be associated with this (x,y) point.
        # First, calc the responsibilities.
        r = np.ones((K), float)
        for k in range(K):
            num = GaussDensity(point.transpose(), centers[k], covarianceMats[k])
            r[k] = mix_coeff[k] * num
        r = r/r.sum() # normalise to get responsibilities
        histo = np.zeros(len(vocabulary),float)
        for k in range(K):
            histo = histo + (r[k] * histograms[k])

        # make a person!
        people.append( Person(x,y,histo) )


    # save the (pickled) people (!), and the vocabulary
    f = open(out_stem + '.slices.pickle', 'w')
    pickle.dump((people,vocabulary),f)
    f.close()



    # make and save a 2D scatter plot
    data = np.zeros((N,2));
    for i,p in enumerate(people):
        data[i,0] = p.x
        data[i,1] = p.y
    pl.scatter(data[:,0], data[:,1], marker='o',s=3,linewidths=None,alpha=1.0)
    pl.axis('equal')
    pl.draw()
    out_imagename = out_stem+'.png'
    pl.savefig(out_imagename)
    print 'saved image ',out_imagename


    for word in vocabulary:
        print 'WORD: ',word
        ind = vocabulary.index(word)
        print 'prevalence in the original histograms: '
        for k in range(K):
            print('%.2f ' % ((histograms[k])[ind])),
        print ' '
        for i,p in enumerate(people[0:3]):
            print '\te.g. intensity for person %d is %.2f' % (i,p.histo[ind])
            
        # make a 2D scatter plot reflecting JUST THIS WORD
        data = np.zeros((N,2))
        intensity = np.zeros(N)
        max_intensity = 0.0
        for i,p in enumerate(people):
            data[i,0] = p.x
            data[i,1] = p.y
            intensity[i] = p.histo[ind]
            if (intensity[i] > max_intensity): max_intensity = intensity[i]
        mycolours = []
        for i,p in enumerate(people):
            # colour fades to white as intensity decreases to zero.
            mycolours.append(np.array([1,1,1]) * (1-p.histo[ind]/max_intensity))
        pl.scatter(data[:,0], data[:,1], c=mycolours, marker='o',s=3,linewidths=None,edgecolors=mycolours,alpha=1.0)
        pl.axis('equal')
        pl.draw()
        out_imagename = out_stem+'_' + word + '.png'
        pl.savefig(out_imagename)
        print 'saved image ',out_imagename


