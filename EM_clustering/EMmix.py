"""
EMmix.py - cluster data using a mixture of Gaussians generative model.
Marcus Frean

Expects to read a datafile consisting of a matrix in which each row is
a training item.
"""

import sys, math
import pylab as pl
import numpy as np
import numpy.random as rng
import numpy.linalg as linalg
from matplotlib.patches import Ellipse


def GaussianDensity(d,mu,v):
    """
    returns the prob of data d (matrix with items as rows), under the
    Gaussian given by mu (vector) and var (matrix)
    """
    normalisation = np.power(2*math.pi,len(mu)/2) * math.sqrt(linalg.det(v))
    xx = d-mu
    yy = np.dot(linalg.inv(v), xx.transpose()).transpose()
    s1 = xx[:,0] * yy[:,0]
    s2 = xx[:,1] * yy[:,1]
    u = -0.5*(s1+s2)
    p = np.exp(u)/normalisation 
    return p


def plotEllipse(pos,P,edge,face,transparency):
    U, s , Vh = pl.svd(P)
    orient = math.atan2(U[1,0],U[0,0])*180/math.pi
    ellipsePlot = Ellipse(xy=pos, width=2.0*math.sqrt(s[0]),
                          height=2.0*math.sqrt(s[1]), angle=orient,
                          facecolor=face,edgecolor=edge,alpha=transparency, 
                          zorder=10)

    ax = pl.gca()
    ax.add_patch(ellipsePlot)
    return ellipsePlot


def learnMOGmodel(data, model):
    (means, variances, mix_coeff) = model
    (N,D) = data.shape
    (D,K) = means.shape
    r = 1.0* np.ones((N,K))                   # start off all the same
    for iteration in range(20):
        # E step___________________________________________________
        # Evaluate the responsibilities using the current parameter
        # values.
        for k in range(K):
            r[:,k] = mix_coeff[k] * GaussianDensity(data, means[:,k], variances[:,:,k])
        r_sum = r.sum(1)

        # gamma is 'responsibility', or r normalised, as per Bishop
        gamma = (r.transpose() / r_sum).transpose()
        gamma_sums = gamma.sum(0) #ie. summed over the 0-th dimension

        # M step___________________________________________________
        for k in range(K):
            g = (data.transpose() * gamma[:,k]).sum(1)
            # update the means
            means[:,k] = g / gamma_sums[k]
            # update the mixing coefficient
            mix_coeff[k] = gamma_sums[k]/sum(gamma_sums)
            # update the (co)variances
            x = data - means[:,k]
            v00 = sum(x[:,0]*x[:,0] * gamma[:,k])
            v01 = sum(x[:,0]*x[:,1] * gamma[:,k])
            v10 = sum(x[:,1]*x[:,0] * gamma[:,k])
            v11 = sum(x[:,1]*x[:,1] * gamma[:,k])
            # Would be a good idea to include SOMETHING LIKE v00 =
            # min(v00,0.001) etc to prevent variances collapsing to
            # zero........
            variances[:,:,k] = [[v00,v01],[v10,v11]] / gamma_sums[k]

        logL = np.log(r_sum).sum() # that was easy!
        #print 'iteration %3d logL %12.6f' % (iteration,logL)
    # END OF THE EM LOOP____________________________________________
    model = (means, variances, mix_coeff)
    return model, logL


def randomStartPoint(N,D,K,data):
    # Set initial guestimates for means, variances, and mixture coefficients
    means = np.zeros((D,K),float)   # centers start off
    for k in range(K):           # on randomly chosen data points
        n = rng.randint(N)
        means[:,k] = data[n,:]
    variances = 10*np.ones((D,D,K),float)     # initial variances
    for k in range(K):                     # start off spherical
        variances[:,:,k] = np.eye(2)          
    mix_coeff = np.ones((K),float)/K          # mixing coefficients
    model = (means, variances, mix_coeff)
    return model 

def findBestMOGmodel(N,D,K,data):
    best_logL = -10000000.0
    for trial in range(20):
        model = randomStartPoint(N,D,K,data)
        m, logL = learnMOGmodel(data, model)
        print 'logL is ',logL
        if logL > best_logL:
            best_logL = logL
            model = m
    print 'best logL is ',best_logL
    return model

#-----------------------------------------------------------------------------

if __name__ == '__main__':

    # Set the random number generator's seed if you want reproducability.
    # (nb. this would be better as a command line arg...)
    # rng.seed(1112)  

    if len(sys.argv) == 3:
        K = int(sys.argv[1])
        infile = str(sys.argv[2])
        print 'We are assuming %d classes' % (K)
    else:
        sys.exit('usage: python EMmix.py numClasses infile')

    data = np.genfromtxt(infile, float) #, unpack=True)
    (N,D) = data.shape
    print 'N is ',N, ' and D is ',D


    (means, variances, mix_coeff) = findBestMOGmodel(N,D,K,data)


    f1 = pl.figure()
    pl.title('%d Gaussians fit to %s using EM' % (K, infile))
    randColor = np.array([.2,.75, 1])
    for k in range(K):
        randColor = rng.random((3))
        randColor /= randColor.sum()
        ellipsePlot=plotEllipse(means[:,k],variances[:,:,k],'blue',
                                randColor,mix_coeff[k]/mix_coeff.max())

    pl.scatter(data[:,0], data[:,1], marker='o',s=.5,linewidths=None,alpha=0.5)
    pl.axis('equal')  # uncomment to give the 2 axes the same scale
    pl.draw()

    out_stem = infile.split('.')[0]
    out_image = out_stem + '_EM.png'
    pl.savefig(out_image)
    print '\n  saved image ',out_image


    # for fun, we can now show samples from this model, to cf. original data.
    f2 = pl.figure()
    pl.title('What the model captures')
    X = np.zeros((N,D))
    for i in range(N):
        j = np.sum(rng.random() > np.cumsum(mix_coeff))
        X[i,:] = np.array([rng.multivariate_normal(means[:,j],variances[:,:,j],1)])
    pl.scatter(X[:,0], X[:,1], marker='o',s=.5,linewidths=None,alpha=0.5)
    pl.axis('equal')
    pl.draw()
    out_imagename = out_stem+'_faked.png'
    pl.savefig(out_imagename)
    print '\n  saved image ',out_imagename
