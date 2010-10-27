import numpy
import theano
import theano.tensor as T

initial_W = numpy.asarray( [[0.1,0.2,0.3], \
                            [0.1,0.2,0.3], \
                            [0.1,0.2,0.3]], \
                            dtype = theano.config.floatX)
W = theano.shared(value = initial_W, name = 'W')
vbias=theano.shared(value=0.1, name='vbias') #0.01


def propdown(hid):
    pre_sigmoid_activation = T.dot(hid, W.T) + vbias
    return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]

def propdown_softmax(hid):
    pre_sigmoid_activation = T.nnet.softmax(T.dot(hid, W.T) + vbias)
    return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]


hid=T.matrix('hid')
pre_activation1, output = propdown(hid)
f = theano.function([hid], output)

pre_activation1, output = propdown_softmax(hid)
f_softmax = theano.function([hid], output)


#pre_sigmoid_activation = T.dot(hid, W.T) + vbias
#f = theano.function([hid], pre_sigmoid_activation)
print f([[0,1,0],[0,0,1]]) #works fine of course

#pre_sigmoid_activation2 = T.nnet.softmax(T.dot(hid, W.T) + vbias)
#f_softmax = theano.function([hid], pre_sigmoid_activation2)
#BUG "ERROR: Optimization failure due to: local_softmax_with_bias" occurs here.. 
print f_softmax([[0,1,0],[0,0,1]])
