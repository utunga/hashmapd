"""
Read in the records of 'people', each consisting of an (x,y) position
and a histogram over words.

Use EM to fit a mixture of Gaussians model to the (x,y) data.

For each word, we want to calculate OVERALL responsibilities (?)
across the mixture components, and then consider the Kullback-Liebler
divergence between these and the (background) mixture coefficients.
Words with the highest KL divergence have distributions that are the
most different from the overall density, hence pointing them out is
'saying something'.

Where to point them out?  KL is a sum over components: look for the
largest (1,2,3?) contributions to this sum.

MY BIG WORRY: say we learn 2 nearby Gaussians instead of 1, for some
cluster. Will this be treated differently from if we'd caught the one?
If so that's a bad thing. I've looked cursorily at ideas for merging
components, but not enough to solve this yet. Merging smacks of
agglomerative hierarchical clustering, which stinks...

Here are logL for best case EM runs, with different K.  Ground truth
was that 'testpeople' was made with 5 components:

K  logL
1  -5332
2  -4597
3  -4513
4  -4473
5  -4453
6  -4449
7  -4444
8  -4444
9  -4439
10 -4435

ie. there's a bit of an 'elbow' at about the right point. I've seen
this before: it may be sufficent to detect the elbow point, and use
that.  SO... how to detect elbows?! Not easy, as EM results are
stochastic, and the elbow may be as smooth as it wants to be.

Also, do you even get the 'elbow' if the data aren't from a MoG in the
first place? Likely to be even less obvious what 'K should be'. K too
low means we may miss stuff. K too high means we may miss IMPORTANT
stuff owing to failing to merge. So we should either implement merging
or aim to set K on the low side, somehow.

Marcus.
"""

import sys, pickle
import pylab as pl
import numpy as np
import numpy.random as rng
from makeMixture import makeRandom2DCovMatrix
from makeFakePeople import Person
from EMmix import *


#-----------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) == 3:
        in_file = str(sys.argv[1])
        out_stem = in_file.split('.')[0]
        K = int(sys.argv[2])  # the number of clusters EM will use
    else:
        sys.exit('usage: python   labelFakePeople.py   filename  num_clusters')
        

    f = open(in_file,'r')
    (people,vocabulary) = pickle.load(f)
    f.close()

    for p in people[:4]:
        print('person %3d is at %.2f,%.2f' % (people.index(p), p.x, p.y))
        print('\t histogram: '),
        print(p.histo)
    data = np.zeros((len(people),2),float)
    for i,p in enumerate(people):
        data[i,:] = [p.x,p.y]
    

    # learn a decent mixture of Gaussians model for the overall density.
    (means, variances, mix_coeff) = findBestMOGmodel(len(people),2,K,data)

    # make a plot
    f1 = pl.figure()
    pl.title('%d Gaussians fit to %s using EM' % (K, out_stem))
    ellipseColor = np.array([.2,.75, 1])
    for k in range(K):
        ellipsePlot=plotEllipse(means[:,k],variances[:,:,k],'blue',
                                ellipseColor,mix_coeff[k]/mix_coeff.max())
    pl.scatter(data[:,0], data[:,1], marker='o',s=3,linewidths=None,alpha=0.5)
    pl.axis('equal')  # uncomment to give the 2 axes the same scale
    pl.draw()
    out_image = out_stem + '_EM.png'
    pl.savefig(out_image)
    print '\n  saved image ',out_image

    

    """
    OR, here's an ALTERNATIVE idea.................

    For each word, learn a MoG model that takes the WEIGHTED data
    points as its input. Store these models in a list. Then:

    The decision about whether to display the word at all will be
    based on the (i) overall frequency of the word, (ii) the entropy
    of its use. I think we can probably just calculate the entropy of
    the mixture distribution itself directly. ??

    IF the word is frequent enough, and its entropy is high (eg. just
    rank them and take the top few?), the POSITIONS to display it
    should be those for which its density is anomalously high,
    presumably. That will be determined the mixture coefficients, and
    the covariance matrix - we are interested in not-too-fat and
    not-too-thin? Need to think through the most principled way to do
    this.
    """
