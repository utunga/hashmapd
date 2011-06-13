""" This file contains different utility functions that are not connected 
in anyway to the networks presented in the tutorials, but rather help in 
processing the outputs into a more understandable way. 

For example ``tiled_array_image`` helps in generating a easy to grasp 
image from a set of samples or weights.
"""


import numpy, PIL.Image


def tiled_array_image(A):
    """
    Transform an array with one flattened image per row, into an image in 
    which the rows are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    """
 
    # Turn [tiles, pixels] array 'A' into
    # [X, Y] tiles each being [x,y] pixels.
    # where X*Y = tiles and x*y = pixels
    # in an [X*x, Y*y] image, plus padding.
    
    A = A.T
    (tiles, pixels) = A.shape
    if tiles >= pixels:
        A = A.T
        (tiles, pixels) = (pixels, tiles)
    x = int(numpy.sqrt(pixels))
    y = (pixels-1) // x + 1
    img_shape = (x, y)
    pad = 1 if max(x,y) < 4 else 2
    tile_spacing = (pad, pad)
    (x1, y1) = [d+s for (d,s) in zip(img_shape, tile_spacing)]
    X = min((300 // x1) + 1, int(numpy.sqrt(tiles))+1)
    Y = min((300 // y1) + 1, (tiles-1) // X + 1)
    tile_shape = (X, Y)

    # scale_rows_to_unit_interval
    A = A.copy()
    A -= A.min(axis=1)[:, numpy.newaxis]
    A /= (A.max(axis=1)[:, numpy.newaxis] + 1e-8)
    
    # pad because pixels per tile may be not square
    padded = numpy.zeros([A.shape[0], numpy.product(img_shape)], A.dtype)
    padded[:] = numpy.NaN
    padded[:, 0:A.shape[1]] = A
    A = padded

    (xp, yp) = x+pad, y+pad
    out_array = numpy.zeros([X*xp-pad, Y*yp-pad], dtype=A.dtype)
    out_array[:] = numpy.NaN

    for tile in range(min(X*Y, tiles)):
        (row, col) = divmod(tile, X)
        out_array[row*xp:row*xp+x, col*yp:col*yp+y] = A[tile].reshape(img_shape)

    image = PIL.Image.fromarray(255*numpy.nan_to_num(out_array)).convert("L")
    mask = PIL.Image.fromarray(255-255*(numpy.isnan(out_array))).convert("L")
    image.putalpha(mask)
    scale = 300 // max(image.size)
    if scale > 1:
        scale = min(scale, 10)
        image = image.resize([scale*d for d in image.size])

    return image



