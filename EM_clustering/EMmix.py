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
                          zorder=0)

    ax = pl.gca()
    ax.add_patch(ellipsePlot)
    return ellipsePlot


def learnMOGmodel(data, weightings, model):
    """
    weightings allows us to give different relative importance to
    different data items. In the 'vanilla' use of EMmix, the
    weightings should all be one.
    """
    print 'EM weightings : ',
    print weightings[0:5]
    (means, variances, mix_coeff) = model
    (N,D) = data.shape
    (D,K) = means.shape
    r = 1.0* np.ones((N,K))                   # start off all the same
    for iteration in range(100):
        # E step___________________________________________________
        # Evaluate the responsibilities using the current parameter
        # values.
        for k in range(K):
            r[:,k] = mix_coeff[k] *GaussianDensity(data, means[:,k], variances[:,:,k])
        r_sum = r.sum(1)



        # Question: how to make EM feel the weightings?
        r_sum = r_sum * weightings # ***** THIS IS NOT CORRECT YET ! *****
        # WARNING: these weightings, done naively here, will screw up logL.




        
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


def findBestMOGmodel(N,D,K,data,weightings):
    best_logL = -10000000.0
    for trial in range(1):  # do several trials and pick the best.
        model = randomStartPoint(N,D,K,data)
        m, logL = learnMOGmodel(data,weightings, model)
        print 'logL is ',logL
        if logL > best_logL:
            best_logL = logL
            model = m
    print 'best logL is ',best_logL
    return model


def randomStartPoint(N,D,K,data):
    # Set initial guestimates for means, variances, and mixture coefficients
    means = np.zeros((D,K),float)   
    for k in range(K):           # centers on randomly chosen data points
        n = rng.randint(N)
        means[:,k] = data[n,:]
    variances = 10*np.ones((D,D,K),float)  
    for k in range(K):           # variances start off equal / spherical
        variances[:,:,k] = np.eye(2)          
    mix_coeff = np.ones((K),float)/K    # mixing coefficients are uniform
    model = (means, variances, mix_coeff)
    return model 

#-----------------------------------------------------------------------------

if __name__ == '__main__':

    # Set the random number generator's seed if you want reproducability.
    # (nb. this would be better as a command line arg...)
    # rng.seed(1112)  

    if len(sys.argv) == 3:
        infile = str(sys.argv[1])
        K = int(sys.argv[2])
    else:
        sys.exit('usage: python EMmix.py  infile num_Classes ')

    data = np.genfromtxt(infile, float) #, unpack=True)
    (N,D) = data.shape
    print 'N is ',N, ' and D is ',D


    (means, variances, mix_coeff) = findBestMOGmodel(N,D,K,data,np.ones(N))


    f1 = pl.figure()
    pl.title('%d Gaussians fit to %s using EM' % (K, infile))
    randColor = np.array([.6,1.0, 1.0])
    for k in range(K):
        randColor = rng.random((3))
        randColor /= randColor.sum()
        mu = means[:,k]
        cov = variances[:,:,k]
        ellipsePlot=plotEllipse(mu,cov,'blue',randColor,mix_coeff[k]/mix_coeff.max())

        # stuff to put words on the clusters
        u,s,vh = linalg.svd(cov)
        princ_comp = u[0]
        print 'principal_component is ',princ_comp
        angle = (360/(2*math.pi)) * math.atan2(princ_comp[1],princ_comp[0])
        if (angle>90):  angle = angle-180 # we don't like upside-down text
        if (angle<-90): angle = angle+180
        # overrule text orientation if there's not that much diff?
        # s ratio tells us relative dominance of 1st vs 2nd principal component:
        if (s[0]/s[1] < 4.0): angle = 0.0
        labelText = 'component ' + str(k)
        labelSize = 15*mix_coeff[k]/max(mix_coeff)
        if labelSize > 5:
            pl.text(mu[0],mu[1],labelText,size=labelSize,rotation=angle,ha='center',va='center',alpha=1.0)


    pl.scatter(data[:,0], data[:,1], marker='o',s=1.0,linewidths=None,alpha=0.1)
    pl.axis('equal')  # uncomment to give the 2 axes the same scale
    pl.draw()

    out_stem = infile.split('.')[0]
    out_image = out_stem + '_EM.png'
    pl.savefig(out_image)
    print '\n  saved image ',out_image


    # for fun, we can now show samples from this model, to cf. original data.
    """
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
    """
