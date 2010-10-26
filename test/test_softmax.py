import numpy
import theano
import theano.tensor as T

initial_W = numpy.asarray( [[0.1,0.1,0.1], \
                            [0.1,0.1,0.1], \
                            [0.1,0.1,0.1]], \
                            dtype = theano.config.floatX)
W = theano.shared(value = initial_W, name = 'W')
vbias=theano.shared(value=0.1, name='vbias') #0.01

hid=T.vector('hid')
pre_sigmoid_activation = T.dot(hid, W.T) + vbias
f = theano.function([hid], pre_sigmoid_activation)
print f([0,1,0]) #works fine of course

pre_sigmoid_activation2 = T.nnet.softmax(T.dot(hid, W.T) + vbias)[0]
f_softmax = theano.function([hid], pre_sigmoid_activation2)
#BUG "ERROR: Optimization failure due to: local_softmax_with_bias" occurs here.. 
print f_softmax([0,1,0])

