from theano import function, scan
import theano.tensor as T
import numpy as np

def inner(i, square, j, vec1, vec2):
    cost = T.abs_(vec1[i] - vec2[j])
    stack = T.stack(square[i, j+1], square[i+1, j], square[i, j])
    dist = cost + T.min(stack)
    square = T.set_subtensor(square[i+1, j+1], dist)

    return square


def outer(j, square, inner_loop, vec1, vec2):
    cost, _ = scan(fn=inner,
                    outputs_info=[dict(initial=square, taps=[-1])],
                    non_sequences=[j, vec1, vec2],
                    sequences=inner_loop)
    return cost[-1]


def dtw(array1, array2):
    s = np.zeros((array1.size+1, array2.size+1))

    s[:,0] = 100
    s[0,:] = 100
    s[0,0] = 0.0

    square = T.dmatrix('square')
    vec1 = T.dvector('vec1')
    vec2 = T.dvector('vec2')
    vec1_length = T.dscalar('vec1_length')
    vec2_length = T.dscalar('vec2_length')
    outer_loop = T.arange(vec1_length, dtype='int64')
    inner_loop = T.arange(vec2_length, dtype='int64')

    cost, _ = scan(fn=outer,
                    outputs_info=[dict(initial=square, taps=[-1])],
                    non_sequences=[inner_loop, vec1, vec2],
                    sequences=outer_loop)

    theano_square = function([vec1, vec2, square, vec1_length, vec2_length], cost, on_unused_input='warn')

    return theano_square(array1, array2, s, array1.size, array2.size)


if __name__=="__main__":
    array1 = np.array([1.0,1.0,5.0,1.0,1.0])
    array2 = np.array([1.0,1.0,1.0,6.0,1.0])
    print dtw(array1, array2)
