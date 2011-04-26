import sys
import csv
import numpy as np
from pylab import plot, draw, figure, imshow, xlabel, ylabel, cm, show, axis, savefig, text
from scipy import stats, mgrid, c_, reshape, random, rot90
from csv_unicode_helpers import UnicodeReader


class Render(object):
    def __init__(self, coords_filename, labels_filename = None):
        self.coords_filename = coords_filename
        self.has_labels = False
        if (labels_filename != None):
            self.has_labels = True
            self.labels_filename = labels_filename
        self.fig_width  = 8
        self.fig_height = 8


    def plot_density(self, plot_filename='out/density.png'):
        x,y,labels = self.load_data()

        figure(figsize=(self.fig_width,self.fig_height), dpi=80)
        # Perform a kernel density estimator on the coords in data.
        # The following 10 lines can be commented out if density map not needed.
        space_factor = 1.2
        xmin = space_factor*x.min()
        xmax = space_factor*x.max()
        ymin = space_factor*y.min()
        ymax = space_factor*y.max()
        X, Y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = c_[X.ravel(), Y.ravel()]
        values = c_[x, y]
        kernel = stats.kde.gaussian_kde(values.T)
        Z = reshape(kernel(positions.T).T, X.T.shape)
        imshow(rot90(Z), cmap=cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])


        # Plot the twits
        if self.has_labels:
            for i in range(len(labels)):
                text(x[i],y[i],labels[i]) # assumes m size and order matches labels
        else:
            plot(x, y, 'k.', markersize=1)
        axis('equal')
        axis('off')
        savefig(plot_filename)
        print 'wrote %s' % (plot_filename)


    def load_data(self):
        
        labels = []
        if self.has_labels:
            unicodeReader = UnicodeReader(open(self.labels_filename,'r'))
            for row in unicodeReader:
                labels.append(row[0])

        data = np.genfromtxt(self.coords_filename, delimiter=",") # loads the coords into 'data'

        x = data[:,0] # x-coords
        y = data[:,1] # y-coords
        return x,y,labels
        


#------------------------------


if __name__ == '__main__':


    if len(sys.argv) < 2:
        sys.exit("usage: python %s  coordsfile.csv   [labelsfile.csv]" % (sys.argv[0]))

    coords_file = sys.argv[1]
    file_stem = coords_file.rstrip('.csv')
