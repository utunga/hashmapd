#! /usr/bin/env python
"""
Python wrapper to execute c++ tSNE implementation
for more information on tSNE, go to :
http://ticc.uvt.nl/~lvdrmaaten/Laurens_van_der_Maaten/t-SNE.html

HOW TO USE
Just call the method calc_tsne(dataMatrix)

Created by Philippe Hamel
hamelphi@iro.umontreal.ca
October 24th 2008

Bits of code hacked onto this by MKT 2010-09

Yes it needs to be modularized badly!
"""

from struct import *
import sys
import os
from numpy import *
import csv



# Some fileconstants

#real data
#WORD_VECTORS_FILE = "../data/user_word_vectors.csv";
#WORD_VECTORS_NUM_WORDS = 3000; #number of different words in above file ('word_id' column is allowed to be *up to and including* this number)
#WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)

#test data
WORD_VECTORS_FILE = "data/test_word_vectors.csv";
WORD_VECTORS_NUM_WORDS = 500; #500 words for testing.. (speeds stuff up a bit)
WORD_VECTORS_NUM_USERS = 786; #number of users for which we have data in above file ('user_id' column is allowed to be *up to and including* this number)



INTERIM_PCA_DIMS = 200; #number of dimensions to reduce as interim step
COORDS_OUTPUT_FILE = "out/coords.csv"; 

def calc_tsne(dataMatrix,NO_DIMS=2,PERPLEX=30,INITIAL_DIMS=30,LANDMARKS=1):
    """
    This is the main function.
    dataMatrix is a 2D numpy array containing your data (each row is a data point)
    Remark : LANDMARKS is a ratio (0<LANDMARKS<=1)
    If LANDMARKS == 1 , it returns the list of points in the same order as the input
    """
    
    dataMatrix=PCA(dataMatrix,INITIAL_DIMS)
    writeDat(dataMatrix,NO_DIMS,PERPLEX,LANDMARKS)
    tSNE()
    Xmat,LM,costs=readResult()
    clearData()
    if LANDMARKS==1:
        X=reOrder(Xmat,LM)
        return X
    return Xmat,LM

def PCA(dataMatrix, INITIAL_DIMS) :
    """
    Performs PCA on data.
    Reduces the dimensionality to INITIAL_DIMS
    """
    print 'Performing PCA'

    dataMatrix= dataMatrix-dataMatrix.mean(axis=0)

    if dataMatrix.shape[1]<INITIAL_DIMS:
        INITIAL_DIMS=dataMatrix.shape[1]

    (eigValues,eigVectors)=linalg.eig(cov(dataMatrix.T))
    perm=argsort(-eigValues)
    eigVectors=eigVectors[:,perm[0:INITIAL_DIMS]]
    dataMatrix=dot(dataMatrix,eigVectors)
    return dataMatrix

def readbin(type,file) :
    """
    used to read binary data from a file
    """
    return unpack(type,file.read(calcsize(type)))

def writeDat(dataMatrix,NO_DIMS,PERPLEX,LANDMARKS):
    """
    Generates data.dat
    """
    print 'Writing data.dat'
    print 'Dimension of projection : %i \nPerplexity : %i \nLandmarks(ratio) : %f'%(NO_DIMS,PERPLEX,LANDMARKS)
    n,d = dataMatrix.shape
    f = open('data.dat', 'wb')
    f.write(pack('=iiid',n,d,NO_DIMS,PERPLEX))
    f.write(pack('=d',LANDMARKS))
    for inst in dataMatrix :
        for el in inst :
            f.write(pack('=d',el))
    f.close()


def tSNE():
    """
    Calls the tsne c++ implementation depending on the platform
    """
    platform=sys.platform
    print'Platform detected : %s'%platform
    if platform in ['mac', 'darwin'] :
        cmd='lib/tSNE_maci'
    elif platform == 'win32' :
        cmd='lib/tSNE_win'
    elif platform == 'linux2' :
        cmd='lib/tSNE_linux'
    else :
        print 'Not sure about the platform, we will try linux version...'
        cmd='lib/tSNE_linux'
    print 'Calling executable "%s"'%cmd
    os.system(cmd)
    

def readResult():
    """
    Reads result from result.dat
    """
    print 'Reading result.dat'
    f=open('result.dat','rb')
    n,ND=readbin('ii',f)
    Xmat=empty((n,ND))
    for i in range(n):
        for j in range(ND):
            Xmat[i,j]=readbin('d',f)[0]
    LM=readbin('%ii'%n,f)
    costs=readbin('%id'%n,f)
    f.close()
    return (Xmat,LM,costs)

def reOrder(Xmat, LM):
    """
    Re-order the data in the original order
    Call only if LANDMARKS==1
    """
    print 'Reordering results'
    X=zeros(Xmat.shape)
    for i,lm in enumerate(LM):
        X[lm]=Xmat[i]
    return X

def clearData():
    """
    Clears files data.dat and result.dat
    """
    print 'Clearing data.dat and result.dat'
    os.system('rm data.dat')
    os.system('rm result.dat')

def writeCoords(mappedX):
    """
    Writes out coordinates data to a csv file
    """
    print 'Writing ' + COORDS_OUTPUT_FILE
    n,d = mappedX.shape
    f = open(COORDS_OUTPUT_FILE, 'wb')
    for row in mappedX :
        f.write(format(row[0],'f'))
        f.write(',')
        f.write(format(row[1],'f'))
        f.write('\n');
    f.close()

def readUserWordCounts(input_rows, input_dims):
    """
    Reads in data from user_word_vectors.csv
    """
    print "attempting to read " + WORD_VECTORS_FILE
    
    numpyArray = zeros((input_rows, input_dims), dtype=int16);
    
    vectorReader = csv.DictReader(open(WORD_VECTORS_FILE, 'rb'), delimiter=',')
    count =0;
    for row in vectorReader:
        if count % 10000==0:
            print 'reading row '+ str(count) + '..'; #MKT: presumably a nicer way to do this
        count += 1;
        inp_row = int(row['user_id'])-1;
        inp_col = int(row['word_id'])-1;
        numpyArray[inp_row,inp_col] = row['count'];
    
    print 'done reading input';
    print numpyArray;
    return numpyArray;
        
        
if __name__ == '__main__':  

    #no_dims = 2;
    #init_dims = 30;
    #perplexity = 30;
    #X = array([[1,2,3],[2,3,4]]);
    #mappedX = calc_tsne(X);
    #print mappedX;
    #writeCoords(mappedX);
    
    input_dims = WORD_VECTORS_NUM_WORDS; 
    #input_dims = 500; #500 words for testing
    input_rows = WORD_VECTORS_NUM_USERS; 
    
    inputX = readUserWordCounts(input_rows, input_dims);

    desired_dims = 2;
    pca_dims = 200;
    perplexity=5;
    mappedX = calc_tsne(inputX, desired_dims, perplexity, pca_dims);
    print mappedX;
    writeCoords(mappedX);