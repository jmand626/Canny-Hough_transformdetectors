
import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Flip kernel both vertically and horizontly
    kernel = np.flip(np.flip(kernel, 0), 1)

    for i in range (Hi):
      for j in range(Wi):
        # Map kernel onto image with kernel by figuring out current region
        mapped = padded[i: i + Hk, j: j + Wk]
        out[i, j] = np.sum(mapped *kernel)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Not very clear, but we were supposed to calculate k in itself.
    # Thankfully, we know that the kernel must be of the size 2k+1 and
    # we know what size is
    k = (size - 1) / 2

    for i in range(size):
      for j in range(size):
        kernel[i, j] = (1/(2 * np.pi * sigma ** 2)) * np.exp(-((i - k) 
        ** 2 + (j-k) ** 2)/(2 * sigma ** 2))

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # We derived formulas for derivatives in lectures, but we can use
    # the expected outputs listed in the cells them selves because the
    # test starts with something similar to the identity matrix (1 in center)
    out = conv(img, np.array([[ 0, 0, 0],[ 0.5, 0, -0.5],[ 0, 0, 0]]))
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = conv(img, np.array([[ 0, 0.5, 0],[ 0, 0, 0],[ 0, -0.5, 0]]))
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    g_x = partial_x(img)
    g_y = partial_y(img)

    G = np.sqrt(g_x ** 2 + g_y ** 2)
    theta = np.arctan2(g_y, g_x)
    # One important piece seen by the message is that arctan2 returns
    # the degree in radians, so we have to convert that.
    theta = np.degrees(theta)

    # That still did not ensure that theta is within the range of 0 to
    # 360, so we do that
    theta %= 360
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    #print(G)
    ### BEGIN YOUR CODE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # For each pixel so we have double for loops

    for i in range(H):
      for j in range(W):
        direction = theta[i, j]
        # We check two pairs (opposite dirs) at a time according to step 2
        pixel1 = 0
        pixel2 = 0

        # For the 8-person neighborhood, we have of course 4 pairs of two
        # each with opposing directions, which is what we want, since this
        # is the comparing with the posistive and negative gradient directions
        # as mentioned

        # Now count up pixels by 45 as mentioned
        if direction == 0 or direction == 180:
          # Parallel to x axis
          if j < W - 1:
            pixel2 = G[i, j + 1]
            # Gradient goes east
          if j > 0:
            pixel1 = G[i, j- 1]
            # Gradient goes west

        elif direction == 45 or direction == 225:
          if i > 0 and j > 0:
            pixel1 = G[i - 1, j - 1]
            # Go northwest
          if i < H - 1 and j< W - 1:
            pixel2 = G[i +1, j + 1]
            # go south east

        elif direction == 90 or direction == 270:
          # Parallel to y axis
          if i < H - 1:
            pixel2 = G[i +1, j]
            # Gradient goes south
          if i > 0:
            pixel1 = G[i - 1, j]
            # Gradient goes north

        elif direction == 135 or direction == 315:
          if i > 0 and j < W-1:
            pixel1 = G[i - 1, j + 1]
            # north east gradient
          if i < H - 1 and j > 0:
            pixel2 = G[i + 1, j-1]
            # south west gradient

        # Now see if the current value is greater than two pixels in 
        # the two directions
        if G[i, j] >= pixel1 and G[i, j] >= pixel2:
          out[i, j] = G[i, j]
        else: 
          out[i, j] = 0


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype= "bool")
    weak_edges = np.zeros(img.shape, dtype= "bool" )

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Although its still read, remember that we can apply booleans to
    # numpy arrays
    strong_edges = img > high
    weak_edges = ((img <= high) & (img > low))
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype= "bool")

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # First of all, notice from the neighbors function that a point is
    # defined with (y, x) not (x, y). We import deque to do BFS
    from collections import deque
    queue = deque()
    for y, x in zip(*np.nonzero(strong_edges)):
      queue.append((y, x))
      # nonzero call to ignore invalid edges, * to unpack, and then zip
      # for an iterator.
    
    while queue: 
      y, x = queue.popleft()
      for ny, nx in get_neighbors(y, x, H, W):
        if weak_edges[ny, nx]:
          # Promote to strong edges because we found iut
          edges[ny, nx] = True
          # as edges is np.copy

          weak_edges[ny, nx] = False
          queue.append((ny, nx))
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 1. Suppress Noise
    k = gaussian_kernel(kernel_size, sigma)
    smooth = conv(img, k)

    # 2. Compute gradient magnitude and direction 
    G, theta = gradient(smooth)

    # 3. Apply Non-Maximum Suppression
    nms = non_maximum_suppression(G, theta)

    # 4. Use hysteresis and connectivity analysis to detect edges
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i, j in zip(ys, xs):
      for idx in range(thetas.shape[0]):
        # Just use the formula given at the top for rho
        rho = j*cos_t[idx] + i*sin_t[idx]
        accumulator[int(rho+diag_len), idx] += 1

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return accumulator, rhos, thetas
