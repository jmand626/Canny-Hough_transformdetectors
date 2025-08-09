
import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Flip kernel right away so we dont forget about it
    kernel = np.flip(kernel, (0, 1))

    # To pad, we pad as mentioned in lecture and oh by dividing the kernel
    # dimensions by half
    pad_h = Hk // 2
    pad_w = Wk // 2

    for i in range(Hi):
      for j in range(Wi):
        # Of course when looking at the formula, f[m, n] is one certain
        # piece that takes two for loops, and to go over all fs, we need
        # two MORE for loops. So, we count the sum of the two loops below
        # here
        sum = 0
        for k in range(Hk):
          for l in range(Wk):
            # So now we have to map the kernel to be centered around the current
            # pixel. We have to subtract off the padding, and then use our k/l
            mapped_i = i - pad_h+k
            mapped_j = j -pad_w + l

            if 0 <= mapped_i < Hi and 0<= mapped_j < Wi:
              sum += kernel[k][l] * image[mapped_i][mapped_j]

            # The code below was how I originally tried to do this, it doesnt
            # work pepehands
            #if i+1 - k >= 0 and i+1 - k < Hi and j+1 - l >= 0 and j+1 - l < Wi:
              # Becaues they said h * j was easier to implement, and clearly
              # now we dont have to deal with any centering/offset issues
              #sum += kernel[k][l] * image[i+1 - k][j+1 - l]
              # Must add a bunch of +1s because we want to align the center
              # of the kernel with the current pixel, but indices start at 0

        out[i][j] = sum
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Well i was going to use np.pad, but a formula is given in the
    # comments above
    out = np.zeros((H+2*pad_height, W+2*pad_width))

    # Set out to be equal to image at pad plus height and width, because
    # out is now pad + height/width + pad, so pad + height/width gets to
    # the end of the nonzero image
    out[pad_height: pad_height + H, pad_width: pad_width + W] = image
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # We obviously want to pad around the border in order to easily do
    # convolution espically on those edges, so we get the center/half,
    # and then pad such that we have half of the current image going out
    # on each sides.
    padded = zero_pad(image, Hk // 2, Wk // 2)
    # Next clearly mentioned above in the comments and other resources
    # / online, convolution is defined by flipping the kernel (where was
    # this ever mentioned in lecture?)
    flip = np.flip(kernel, (0, 1))

    # In the naive version, we started calculating the sum two for loops
    # into the nest of 4, so here we only go down 2 and use np.sum
    for i in range(Hi):
      for j in range(Wi):
        # Now specifically take non padded regions. Padding just makes 
        # calculation not fail, and multipy by kernel flipped
        out[i, j] = np.sum(padded[i: i + Hk, j: j + Wk] * flip)
        # "element-wise multiplication"
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    g = np.flip(g, (0, 1))
    out = conv_fast(f, g)
    # The correlation is like the kernel but without flipping, so since
    # we want to use the conv_fast function but that automatically flips
    # the kernel, so we "pre-flip" it so it gets flipped back to not
    # being flipped.

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Thankfully we are just told what to do here
    g = g - np.mean(g)
    out = cross_correlation(f, g)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Unforunately this is actually as bad as it seems, since we must
    # do this normalizing thing for patches of the total images at a time

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    # Normalize the filter here. Obviously it will not change, so we catch
    # its mean and std here.
    normal_g = (g - np.mean(g)) / np.std(g)

    # I tried to do this without padding and in OH it was mentioned,
    # so here we are.
    padded_f = zero_pad(f, Hg//2, Wg//2)

    for i in range(Hf):
      for j in range(Wf):
        # Now patch-based cross correlation.
        patch = padded_f[i: i + Hg, j : j + Wg]
        # The current amount to worry about

        out[i, j] = np.sum(normal_g * (patch - np.mean(patch)) 
        / np.std(patch))


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out
