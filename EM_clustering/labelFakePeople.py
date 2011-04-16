"""
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

    for wd in vocabulary:
        print wd

    for p in people[:4]:
        print p.x
        
