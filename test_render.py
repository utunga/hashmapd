import sys
import csv
import numpy as np
from pylab import plot, draw, figure, imshow, xlabel, ylabel, cm, show, axis, savefig, text
from scipy import stats, mgrid, c_, reshape, random, rot90
from hashmapd.csv_unicode_helpers import UnicodeReader

if len(sys.argv) < 2:
    sys.exit("usage: python %s  coordsfile.csv   [labelsfile.csv]" % (sys.argv[0]))

coords_file = sys.argv[1]
data = np.genfromtxt(coords_file, delimiter=",") # writes the coords into 'data'

m1 = data[:,0] # x-coords
m2 = data[:,1] # y-coords

# Perform a kernel density estimator on the coords in data.
# The following 10 lines can be commented out if density map not needed.
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()
X, Y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = c_[X.ravel(), Y.ravel()]
values = c_[m1, m2]
kernel = stats.kde.gaussian_kde(values.T)
Z = reshape(kernel(positions.T).T, X.T.shape)
imshow(rot90(Z), cmap=cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])


# Plot the twits
plot(m1, m2, 'k.', markersize=1)
axis('equal')


if len(sys.argv) == 3:
    labels_file = sys.argv[2]
    unicodeReader = UnicodeReader(open(labels_file,'r'))
    labels = []
    for i,row in enumerate(unicodeReader):
        print i,row[0]
        text(m1[i],m2[i],row[0])

savefig('./out/cloud9.png')
show()
