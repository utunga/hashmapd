#  Based on code Created by Laurens van der Maaten on 20-12-08.
#  see point #4

import numpy

import sys
try:
    import psyco
    psyco.full()
    print >> sys.stderr, "psyco is usable!"
except:
    print >> sys.stderr, "No psyco"


class TSNE(object):
        
    def __init__(self, default_iterations = 1000, perplexity = 15, desired_dims=2):
        
        self.default_iterations = default_iterations
        self.perplexity=perplexity
        self.out_dims=desired_dims
        
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.eta = 500
        self.sigma_iterations = 50 # number of iterations to try when finding sigma that matches perplexity *for each row of P, for every iteration* 
        self.min_gain = 0.01
        self.tol = 1e-5
        
    def Hbeta(self, D = numpy.array([]), beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
        
        #print 'HBeta for D, beta ', D, beta
        # Compute P-row and corresponding perplexity
        P = numpy.exp(-D.copy() * beta);
        sumP = sum(P);
        H = numpy.log(sumP) + beta * numpy.sum(D * P) / sumP;
        P = P / sumP;
        return H, P;
        
    def x2p(self):
        """Performs a binary search to get P-values in such a way that each conditional Gaussian has the target perplexity."""

        X = self.codes
        
        # Initialize some variables
        print "Computing pairwise distances..."
        (n, d) = X.shape
        sum_X = numpy.sum(numpy.square(X), 1)
        D = numpy.add(numpy.add(-2 * numpy.dot(X, X.T), sum_X).T, sum_X)
        P = numpy.zeros((n, n))
        sigmas = []
        
        # Loop over all datapoints
        for i in range(n):
        
            # Print progress
            if i % 500 == 0:
                print "Computing P-values for point ", i, " of ", n, "..."
                
            distances_to_i = D[i, numpy.concatenate((numpy.r_[0:i], numpy.r_[i+1:n]))];
            #print distances_to_i
            thisP, sigma = self.get_row_of_P(distances_to_i,self.perplexity)
            #print thisP, sigma
            # Set the row of P we just worked out
            P[i, numpy.concatenate((numpy.r_[0:i], numpy.r_[i+1:n]))] = thisP;
            sigmas.append(sigma)
            
        # Return final P-matrix
        print "Mean value of sigma: ", numpy.mean(sigmas)
        return P;
    
    def get_row_of_P(self, distances, perplexity):
        
        log_perplexity = numpy.log(perplexity)
        
        # Binary search for a value of beta that achieves the required
        # perplexity. Then returns the corresponding P-vector.
        betamin = -numpy.inf; 
        betamax =  numpy.inf;
        beta = 1 #star guess
        
        #print 'computing gaussian kernal'
        # Compute the Gaussian kernel and entropy for the current beta
        (H, thisP) = self.Hbeta(distances, beta) #first guess, probably wrong
    
        #print 'H', H
        #print 'P', thisP
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - log_perplexity
        tries = 0
        while numpy.abs(Hdiff) > self.tol and tries < self.sigma_iterations:
                    
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta
                if betamax == numpy.inf or betamax == -numpy.inf:
                    beta = beta * 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == numpy.inf or betamin == -numpy.inf:
                    beta = beta / 2
                else:
                    beta = (beta + betamin) / 2
                
            # not yet within tolerance - recompute the values
            (H, thisP) = self.Hbeta(distances, beta)
            Hdiff = H - log_perplexity
            tries = tries + 1
            
        sigma = numpy.sqrt(1 / beta)
        return thisP, sigma
    
    def initialize_with_codes(self, codes):
        self.codes = numpy.array(codes)
        (n, d) = self.codes.shape
        self.code_dims = d
        #random initialize the coords
        print '!randomly initializing the coordinates!'
        self.coords = numpy.random.randn(n, self.out_dims)

    def load_from_file(self, coords_file, codes_file):
        codes = numpy.genfromtxt(codes_file, dtype=numpy.float32, delimiter=',')
        coords = numpy.genfromtxt(coords_file, dtype=numpy.float32, delimiter=',')
        if codes.dtype != "float32":
            print "Error: file of codes should have type float32.";
            return -1;
        
        if (codes.shape[0] != coords.shape[0]):
            print "Error: file of codes and coords should have equal number of rows at this point.";
            return -1;
        
        #throway the index in first column (not sure why that is there)
        self.codes = codes[:,1:]
        self.coords = coords
        print 'loaded coords and codes from %s, %s respectively' %(coords_file, codes_file)
        
    def save_coords_to_file(self, coords_file):
        print 'saving coords to %s' %(coords_file)
        numpy.savetxt(coords_file, self.coords, delimiter=',')

    def get_coord_for_code(self, code, iterations = None):
        if (iterations==None):
            iterations = self.default_iterations
         
        # Initialize variables
        X = numpy.vstack(self.codes, code)
        #randomize the start_coord, somewhere towards middle
        y = numpy.random.randn(1, self.out_dims)
        Y.append(self.coords, y,0   )
        
        initial_momentum = 0.5;
        final_momentum = 0.8;
        eta = 500;
        min_gain = 0.01;
        (n, d) = X.shape;
        assert(d==self.code_dims) #these should be the same, right?
        
        dy = numpy.zeros((1, self.out_dims));
        iy = numpy.zeros((1, self.out_dims));
        gain   = numpy.ones((1, self.out_dims));
        
        # work out distances from this point to all other points (in high-d space)
        # FIXME: function below here should be changed to the
        #         proper 1-D case
        sum_X = numpy.sum(numpy.square(X), 1)
        D = numpy.add(numpy.add(-2 * numpy.dot(X, X.T), sum_X).T, sum_X)
        distances = D[-1]

        # i tried the following and even though it has the right
        # shape, it gives the wrong result! ;-( - MKT
        #sum_x = numpy.sum(numpy.square(code),  0)
        #distances = numpy.add(numpy.add(-2 * numpy.dot(code, X.T), sum_x).T, sum_x)

        #get P for 'this code' to all other codes
        thisP = self.get_row_of_P(distances, self.perplexity)
        
        # don't need this any more since we are doing the above - MKT
        #P = self.x2p();
        #P = P + numpy.transpose(P);
        #P = P / numpy.sum(P);
        #P = P * 4;                                    # early exaggeration
        #P = numpy.maximum(P, 1e-12);
        
        # ALL FUNCTIONS BELOW NEED TO BE CHANGED TO THE
        # 1-D CASE (instead of matrix case)
        # Run iterations
        for iter in range(iterations):
            
            # Compute pairwise affinities
            sum_Y = numpy.sum(numpy.square(Y), 1);        
            num = 1 / (1 + numpy.add(numpy.add(-2 * numpy.dot(Y, Y.T), sum_Y).T, sum_Y));
            num[range(n), range(n)] = 0;
            Q = num / numpy.sum(num);
            Q = numpy.maximum(Q, 1e-12);
            
            # Compute gradient
            PQ = P - Q;
            for i in range(n):
                dY[i,:] = numpy.sum(numpy.tile(PQ[:,i] * num[:,i], (self.out_dims, 1)).T * (Y[i,:] - Y), 0);
                
            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
            gains[gains < min_gain] = min_gain;
            iY = momentum * iY - eta * (gains * dY);
            Y = Y + iY;
            Y = Y - numpy.tile(numpy.mean(Y, 0), (n, 1));
            
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                #print "P ,", P
                #print "Q " , Q
                C = numpy.sum(P * numpy.log(P / Q));
                print "Iteration ", (iter + 1), ": error is ", C
                
                
        # update solution
        self.coords = Y;
        

    def fit(self, iterations = None): 
        """Descends along tsne gradient path for specified number of iterations"""

        if (iterations==None):
            iterations = self.default_iterations
        
        if self.codes.dtype != "float32":
            print "Error: array of codes should have type float32.";
            return -1;
        
        # Initialize variables
        X = self.codes
        Y = self.coords
        initial_momentum = 0.5;
        final_momentum = 0.8;
        eta = 500;
        min_gain = 0.01;
        (n, d) = X.shape;
        assert(d==self.code_dims) #these should be the same, right?
        
        dY = numpy.zeros((n, self.out_dims));
        iY = numpy.zeros((n, self.out_dims));
        gains = numpy.ones((n, self.out_dims));
        
        # Compute P-values
        P = self.x2p();
        P = P + numpy.transpose(P);
        P = P / numpy.sum(P);
        P = P * 4;                                    # early exaggeration
        P = numpy.maximum(P, 1e-12);
        
        # Run iterations
        for iter in range(iterations):
            
            # Compute pairwise affinities
            sum_Y = numpy.sum(numpy.square(Y), 1);        
            num = 1 / (1 + numpy.add(numpy.add(-2 * numpy.dot(Y, Y.T), sum_Y).T, sum_Y));
            num[range(n), range(n)] = 0;
            Q = num / numpy.sum(num);
            Q = numpy.maximum(Q, 1e-12);
            
            # Compute gradient
            PQ = P - Q;
            for i in range(n):
                dY[i,:] = numpy.sum(numpy.tile(PQ[:,i] * num[:,i], (self.out_dims, 1)).T * (Y[i,:] - Y), 0);
                
            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
            gains[gains < min_gain] = min_gain;
            iY = momentum * iY - eta * (gains * dY);
            Y = Y + iY;
            Y = Y - numpy.tile(numpy.mean(Y, 0), (n, 1));
            
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = numpy.sum(P * numpy.log(P / Q));
                print "Iteration ", (iter + 1), ": error is ", C
                
            # Stop lying about P-values
            if iter == 100:
                P = P / 4;
                
        # update solution
        self.coords = Y;
                   
#f __name__ == "__main__":
    #print "Run Y = tsne.tsne(X, self.out_dims, perplexity) to perform t-SNE on your dataset."
    #print "Running example on 2,500 MNIST digits..."
    #X = numpy.loadtxt("mnist2500_X.txt");
    #labels = numpy.loadtxt("mnist2500_labels.txt");
    #Y = tsne(X, 2, 50, 20.0);
    #pylab.scatter(Y[:,0], Y[:,1], 20, labels);
