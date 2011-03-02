"""
EMmix.py - cluster data using a mixture of Gaussians generative model.
Marcus Frean

Expects to read a datafile consisting of a matrix in which each row is
a training item.
"""

import sys
from pylab import *
from numpy import *
import numpy.random as rng
import numpy.linalg as linalg
from matplotlib.patches import Ellipse


def GaussianDensity(d,mu,v):
    """
    returns the prob of data d (matrix with items as rows), under the
    Gaussian given by mu (vector) and var (matrix)
    """
    normalisation = pow(2*pi,len(mu)/2) * sqrt(linalg.det(v))
    x = d-mu
    y = dot(linalg.inv(v), x.transpose()).transpose()
    s1 = x[:,0] * y[:,0]
    s2 = x[:,1] * y[:,1]
    u = -0.5*(s1+s2)
    p = exp(u)/normalisation 
    return p

def GaussianDensity_diagonal(d,mu,v):
    """
    THIS version assumes v is diagonal, ie. Gaussian is axis-aligned.
    Returns the prob of data d (matrix with items as rows), under the
    Gaussian given by mu (vector) and var (matrix)
    """
    u = (pow(d[:,0]-mu[0], 2) / v[0,0]) + (pow(d[:,1]-mu[1], 2) / v[1,1])
    p = exp(-u/2) / (2*pi*v[0,0]*v[1,1])
    return p

def plotEllipse(pos,P,edge,face,transparency):
    U, s , Vh = svd(P)
    orient = math.atan2(U[1,0],U[0,0])*180/pi
    ellipsePlot = Ellipse(xy=pos, width=2.0*math.sqrt(s[0]),
                          height=2.0*math.sqrt(s[1]), angle=orient,
                          facecolor=face,edgecolor=edge,alpha=transparency, 
                          zorder=10)

    ax = gca()
    ax.add_patch(ellipsePlot)
    return ellipsePlot


#-----------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) == 3:
        K = int(sys.argv[1])
        infile = str(sys.argv[2])
        print 'Assuming %d classes' % (K)
    else:
        sys.exit('usage: python EMmix.py numClasses infile')

    data = genfromtxt(infile) #, unpack=True)
    (N,D) = data.shape
    print 'N is ',N, ' and D is ',D

    # Initial guestimates for means, variances, and mixing proportions ('mix_coeffs')
    mean = 3 * rng.normal(0.0,1.0,(D,K))   # initial centers
    variance = ones((D,D,K),float)         # initial variances
    for k in range(K):                     # start off spherical
        variance[:,:,k] = eye(2)
    mix_coeff = ones((K),float)/K              # mixing coefficients
    r = 1.0* ones((N,K))


    for iteration in range(100):
        # E step___________________________________________________
        # Evaluate the responsibilities using the current parameter
        # values.
        for k in range(K):
            r[:,k] = mix_coeff[k] * GaussianDensity(data, mean[:,k], variance[:,:,k])
            # Can choose GaussianDensity_diagonal(..) if restricting to
            # axis-aligned Gaussians. But could also just throw away off-diagonals?!
        r_sum = r.sum(1)

        # gamma is 'responsibility', or r normalised, as per Bishop
        gamma = (r.transpose() / r_sum).transpose()
        gamma_sums = gamma.sum(0) #ie. summed over the 0-th dimension

        # M step___________________________________________________
        for k in range(K):
            g = (data.transpose() * gamma[:,k]).sum(1)
            # update the mean
            mean[:,k] = g / gamma_sums[k]
            # update the mixing coefficient
            mix_coeff[k] = gamma_sums[k]/sum(gamma_sums)
            # update the (co)variances
            x = data - mean[:,k]
            v00 = sum(x[:,0]*x[:,0] * gamma[:,k])
            v01 = sum(x[:,0]*x[:,1] * gamma[:,k])
            v10 = sum(x[:,1]*x[:,0] * gamma[:,k])
            v11 = sum(x[:,1]*x[:,1] * gamma[:,k])
            # SOMETHING LIKE v00 = min(v00,0.001) etc to stop singularities.
            variance[:,:,k] = [[v00,v01],[v10,v11]] / gamma_sums[k]

        logL = log(r_sum).sum()
        print 'iteration %3d logL %12.6f' % (iteration,logL)
    # END OF THE EM LOOP____________________________________________


    f1 = figure()
    title('%d Gaussians fit to %s using EM' % (K, infile))
    randColor = array([.2,.75, 1])
    for k in range(K):
        randColor = rng.random((3))
        randColor /= randColor.sum()

        ellipsePlot=plotEllipse(mean[:,k],variance[:,:,k],'blue',
                                randColor,mix_coeff[k]/mix_coeff.max())
        
        #print '\n Mixture component ',k
        #print '\t mean: ', mean[:,k]
        #print '\t variances: ',variance[:,:,k]
        #print '\t mixture coefficient: ',mix_coeff[k]

    scatter(data[:,0], data[:,1], marker='o',s=.5,linewidths=None,alpha=0.5)
    axis('equal')
    draw()

    out_stem = infile.split('.')[0]
    out_image = out_stem + '_EM.png'
    savefig(out_image)
    print 'saved image ',out_image


    # for fun, we can now show samples from this model, to cf. original data.
    f2 = figure()
    title('What the model captures')
    X = zeros((N,D))
    for i in range(N):
        j = sum(rng.random() > cumsum(mix_coeff))
        X[i,:] = array([rng.multivariate_normal(mean[:,j],variance[:,:,j],1)])
    scatter(X[:,0], X[:,1], marker='o',s=.5,linewidths=None,alpha=0.5)
    axis('equal')
    draw()
    out_imagename = out_stem+'_faked.png'
    savefig(out_imagename)
    print 'saved image ',out_imagename
