import numpy, numpy.random
from hashmapd.SMH import SMH

COLS = 3
ROWS = 100

def synthetic_data(rows, noise=0.1):
    ordinal = numpy.arange(1, COLS+1)
    y = numpy.random.randint(2,4,rows)
    data = numpy.remainder.outer(ordinal, y).T == 0
    data = data.astype('float32')
    data *= (1-noise)
    data += numpy.random.uniform(0, noise, data.shape)
    return data

previously_recorded_W = [
    numpy.array([[ 0.24030146, -1.35252476], [ 3.07910728, -1.31781554], [-2.72068667,  2.17062974]]),
    numpy.array([[-4.11487055], [ 2.52171874]]), 
    numpy.array([[-4.4805522 ,  3.22233582]]), 
    numpy.array([[-0.56450444,  4.59026003, -4.0959754 ], [-2.16593575, -2.54363894,  3.37647772]]),
    ]
    
previously_recorded_b = [
    numpy.array([-0.23629102,  0.08483376]),
    numpy.array([ 0.309991]),
    numpy.array([ 1.83637726, -1.01530766]),
    numpy.array([-1.5396179,  -0.6982789,   0.01355893]),
    ]

def test_train():
    numpy.random.seed(1)

    training_data = synthetic_data(ROWS-30)
    validation_data = synthetic_data(30)
    testing_data = synthetic_data(10)
    
    data = (training_data, validation_data, testing_data)
    data = [(a, a.sum(axis=1)[:, numpy.newaxis]) for a in data]
    (training_data, validation_data, testing_data) = data
    (x, x_sums) = training_data

    smh = SMH(
            numpy_rng = numpy.random.RandomState(123),
            mean_doc_size = x.sum(axis=1).mean(), 
            first_layer_type = 'bernoulli', 
            n_ins = x.shape[1],
            mid_layer_sizes = [2],
            inner_code_length = 1,    
    )
    smh.train(training_data, validation_data, testing_data, 
                finetune_lr = 0.3, pretraining_epochs = 100, pretrain_lr = 0.01, training_epochs = 100, 
                method = 'cd', k = 1, noise_std_dev = 0, cost_method = 'squared_diff', 
                batch_size = 2, skip_trace_images=True, weights_file=None)
    
    layers = smh.export_model()

    for ((W,b),expectedW, expectedB) in zip(layers, previously_recorded_W, previously_recorded_b):
        assert numpy.allclose(W.get_value(), expectedW)
        assert numpy.allclose(b.get_value(), expectedB)
    
    #for (W, b) in layers:
    #    print W.get_value()
    
if __name__ == '__main__':
    test_train()