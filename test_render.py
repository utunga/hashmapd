import sys
import csv
import numpy as np
from pylab import plot, draw, figure, imshow, xlabel, ylabel, cm, show, axis, savefig, text
from scipy import stats, mgrid, c_, reshape, random, rot90
from csv_unicode_helpers import UnicodeReader

if len(sys.argv) < 2:
    sys.exit("usage: python %s  coordsfile.csv   [labelsfile.csv]" % (sys.argv[0]))

coords_file = sys.argv[1]
# write the coords into "data"
data = np.genfromtxt(coords_file, delimiter=",")

m1 = data[:,0] # x-coords
m2 = data[:,1] # y-coords
plot(m1, m2, 'k.', markersize=1)
axis('equal')

if len(sys.argv) == 3:
    labels_file = sys.argv[2]
    unicodeReader = UnicodeReader(open(labels_file,'r'))
    #myReader = csv.reader(open(labels_file, 'r'))
    labels = []
    for i,row in enumerate(unicodeReader):
        print i,row
        text(m1[i],m2[i],row[0])
        
show()
savefig('cloud9.png')
