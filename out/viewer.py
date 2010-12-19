"""
Make a density maps from a coords file, and put labels on them from a
labels file.
"""

import sys, csv
import numpy as np


if len(sys.argv) < 2:
    sys.exit("usage: python %s  coords_file.csv   [labels_file.csv]" % (sys.argv[0]))

coords_file = sys.argv[1]
data = np.genfromtxt(coords_file, delimiter=",")



# This uses SciPy's kernel density estimator: http://www.scipy.org/SciPyPackages/Stats
from pylab import plot, draw, figure, imshow, xlabel, ylabel, cm, show, axis, savefig, text
from scipy import stats, mgrid, c_, reshape, random, rot90

#figure(figsize=(20, 10))
m1 = data[:,0]
m2 = data[:,1]
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()


# Perform a kernel density estimator on the coords in data.
X, Y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = c_[X.ravel(), Y.ravel()]
values = c_[m1, m2]
kernel = stats.kde.gaussian_kde(values.T)
Z = reshape(kernel(positions.T).T, X.T.shape)
imshow(rot90(Z), cmap=cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
plot(m1, m2, 'k.', markersize=1)
axis('equal')


if len(sys.argv) == 3:
    labels_file = sys.argv[2]
    #f = open(labels_file, 'r')

    spamReader = csv.reader(open(labels_file, 'rb'), delimiter=' ', quotechar='|')
    labels = []
    for row in spamReader:
        labels.append(row[0])

    for i,wd in enumerate(labels[0:10]):  # just the first few, while debugging.
        print wd
        text(m1[i],m2[i],wd)   # SOME ENCODING PROBLEM HERE STILL....

savefig('cloud.png')
show()

