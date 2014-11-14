from theano import function, config, shared, sandbox, scan
import theano.tensor as T
import numpy as np
#cimport numpy as np
import time


def distance_real(i, square, j):
    cost = T.abs_(t_x[i] - t_y[j])
    sample = T.stack(square[i, j+1], square[i+1, j], square[i, j])
    dist = cost + T.min(sample)
    square = T.set_subtensor(square[i+1, j+1], dist)

    return square


def distance(j, square, seq):
    cost, _ = scan(fn=distance_real,
                    outputs_info=[dict(initial=square, taps=[-1])],
                    non_sequences=j,
                    sequences=seq)
    return cost[-1]


# Compiled
square = T.dmatrix('square')
t_x = T.dvector('x')
t_y = T.dvector('y')
length = T.dscalar('length')
seq = T.arange(length, dtype='int64')
print square
cost, _ = scan(fn=distance,
                outputs_info=[dict(initial=square, taps=[-1])],
                non_sequences=seq,
                sequences=seq)

theano_square = function([t_x, t_y, square, length], cost, on_unused_input='warn')

# Input
n_y = np.array([1.0,1.0,1.0,6.0,1.0])
n_x = np.array([1.0,1.0,5.0,1.0,1.0])
nrows = n_x.size
ncols = n_y.size
s = np.zeros((n_x.size+1, n_y.size+1))

s[:,0] = 100
s[0,:] = 100
s[0,0] = 0.0

print theano_square(n_x, n_y, s, nrows)
