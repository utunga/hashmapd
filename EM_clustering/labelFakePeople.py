"""
Just getting started on this but what the heck.
Marcus.
"""

import sys, pickle
import pylab as pl
import numpy as np
import numpy.random as rng
from makeMixture import makeRandom2DCovMatrix
from makeFakePeople import Person


#-----------------------------------------------------------------------------

if __name__ == '__main__':

    if len(sys.argv) == 2:
        in_file = str(sys.argv[1])
        out_stem = in_file.split('.')[0]
    else:
        sys.exit('usage: python labelFakePeople.py filename')
        

    f = open(in_file,'r')
    (people,vocabulary) = pickle.load(f)
    f.close()

    for p in people[:4]:
        print('person %3d is at %.2f,%.2f' % (people.index(p), p.x, p.y))
        print('\t histogram: '),
        print(p.histo)

    """
    for each word, we will learn a MoG model (several times and pick
    the best) that takes the WEIGHTED data points as its input. Store
    these models in a list. Then:

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
