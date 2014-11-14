from theano import function, scan
import theano.tensor as T
import numpy as np

def inner(i, square, j, vec1, vec2):
    """
    The inner loop that calculates the actual warp path at i,j
    """
    diff = T.abs_(vec1[i] - vec2[j])
    stack = T.stack(square[i, j+1], square[i+1, j], square[i, j])
    distance = diff + T.min(stack)
    square = T.set_subtensor(square[i+1, j+1], distance)

    return square

def outer(j, square, inner_loop, vec1, vec2):
    """
    The outer loop that calculates warp paths per row
    """
    path, _ = scan(fn=inner,
                    outputs_info=[dict(initial=square, taps=[-1])],
                    non_sequences=[j, vec1, vec2],
                    sequences=inner_loop)
    return path[-1]

def dtw(array1, array2):
    """
    Accepts: two one dimensional arrays
    Returns: (float) DTW distance between them.
    """
    s = np.zeros((array1.size+1, array2.size+1))

    s[:,0] = 1e6
    s[0,:] = 1e6
    s[0,0] = 0.0

    # Set up symbolic variables
    square = T.dmatrix('square')
    vec1 = T.dvector('vec1')
    vec2 = T.dvector('vec2')
    vec1_length = T.dscalar('vec1_length')
    vec2_length = T.dscalar('vec2_length')
    outer_loop = T.arange(vec1_length, dtype='int64')
    inner_loop = T.arange(vec2_length, dtype='int64')

    # Run the outer loop
    path, _ = scan(fn=outer,
                    outputs_info=[dict(initial=square, taps=[-1])],
                    non_sequences=[inner_loop, vec1, vec2],
                    sequences=outer_loop)

    # Compile the function
    theano_square = function([vec1, vec2, square, vec1_length, vec2_length], path, on_unused_input='warn')

    # Call the compiled function and return the actual distance
    return theano_square(array1, array2, s, array1.size, array2.size)[-1][array1.size, array2.size]

if __name__=="__main__":
    """
    Example usage: `python dtw.py`
    """
    array1 = np.array([1.0,1.0,5.0,1.0,1.0])
    array2 = np.array([1.0,1.0,1.0,6.0,1.0])
    print dtw(array1, array2)
