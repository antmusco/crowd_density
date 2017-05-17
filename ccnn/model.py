import itertools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ccnn
import util
from   scipy.ndimage.filters import gaussian_filter 


class CcnnModel(object):
    """
    Context Manager class which should be used to make predictions on new
    inputs based on the trained model. A single CcnnModel can be opened for an
    input stream of images of constant size. 
    
    Count predictions for input images are produced by first generating a
    predicted density map for the input image, then integrating over the map to
    produce a count. 
    
    Density maps are generated from an input image by sampling a grid of
    patches across the image and passing all patches these though the trained
    model. The patches are then re-assembled into the output density map - 
    pixels that were sampled by more than one patch are divided by the number
    of patches to compensate for overcounting.
    """
    def __init__(self, cfg, img_shape=[256,256], model_dir=None):
        """
        Initialize the contex manager. Restores the trained model as a graph
        and opens up the managed TensorFlow session.
        Args:
            cfg: Configuration settings for the trained model.
            img_shape: Shape of the input images. Defaults to [256, 256].
        """

        self._cfg       = cfg
        self._model_dir = cfg.TRAIN_CHECKPOINT_DIR if model_dir is None else \
            model_dir
        # The shape of the expected input images, along with the square patch
        # input width and the stride for each patch sample across the input
        # image.
        self._shape  = img_shape
        self._width  = cfg.CNN_INPUT_WIDTH
        self._stride = cfg.CNN_INPUT_WIDTH / 2
        # p_num is the number of patches used along each direction of the
        # overlapped grid: p_num[0] is the number of patches taken across the
        # height of the input image, p_num[1] is the number of patches taken
        # across the width. p_total is simply the product of these, or the
        # total number of patches predicted for a single input image.
        p_num   = [(i / self._stride) - 1 for i in img_shape]
        p_total = p_num[0] * p_num[1]
        # Coordinates along each direction for each patch. The cartesian
        # product between p_coords[0] and p_coords[1] yeilds the upper-left
        # corner of each patch to be sampled from the input image.
        p_coords = [[i * self._stride for i in xrange(c)] for c in p_num]
        self._coords = list(itertools.product(p_coords[0], p_coords[1]))
        # Patch buffer and density buffer- used so we don't have to allocate
        # new arrays for each frame.
        self._p_buf  = np.zeros([p_total, self._width, self._width, 3])
        self._d_buf  = np.zeros(img_shape)
        
        # Construct the divider to account for overlap. The divider is simply a
        # matrix of ints of the same size as the output density map: the value
        # of entry (i,j) represents how many patches made a prediction for
        # pixel (i,j), and thus how much that pixel's value should be divided
        # by.
        ones = np.ones([self._width, self._width])
        self._divider = np.zeros(img_shape)
        for (x, y) in self._coords:
            self._divider[x:x+self._width, y:y+self._width, ...] += ones
        # Make sure we don't divide by zero
        self._divider[self._divider == 0] = 1

        # Build graph.
        self._graph  = tf.Graph()
        with self._graph.as_default():
            self._sess = tf.Session()
            # Insert placeholder to take a batch of patches.
            self._patches = tf.placeholder(tf.float32, shape=(p_total,
                self._width, self._width, 3))
            self._predict = ccnn.inference(self._patches, is_train=True,
                cfg=self._cfg)
            # Load the graph from the last training checkpoint.
            saver = tf.train.Saver()
            ckpt  = tf.train.get_checkpoint_state(self._model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
            else:
                self.__exit__()
                raise Exception('Could not restore training graph.')

    def __enter__(self):
        # Return self as context manager.
        return self

    def __exit__(self, *args):
        # Close the session.
        self._sess.close() 

    def close(self):
        # Close the context manager.
        self.__exit__();
        
    def predict(self, img):
        """
        Generates a density map for the given image, and returns an estimated
        count of the number of objects contained within the view.

        Density maps are generated from an input image by sampling a grid of
        patches across the image and passing all patches these though the trained
        model. The patches are then re-assembled into the output density map - 
        pixels that were sampled by more than one patch are divided by the number
        of patches to compensate for overcounting.

        Args:
            img: Image to predict. Should be same shape as what this model was
                initialized to predict.
        Returns:
            count: Predicted number of objects within 'img'
            dens: Predicted density map of 'img'
        """
        # Make sure we have the correct size.
        if any(i != j for i, j in zip(img.shape[:2], self._shape)):
            raise Exception('Expected shape {}, got {}'.format(self._shape,
                img.shape[:2]))

        # Extract each patch from source image.
        for i, (x, y) in enumerate(self._coords):
            self._p_buf[i, ...] = img[x:x+self._width, y:y+self._width, ...] 

        # Pass patches through CCNN.
        with self._graph.as_default():
            d_predict = self._sess.run(self._predict, feed_dict={self._patches:
                self._p_buf})
            d_predict = d_predict * 0.25;

        # Reconstruct density map.
        patch_width = (self._width, self._width)
        self._d_buf = np.zeros(self._shape)
        for i, (x, y) in enumerate(self._coords):
            self._d_buf[x:x+self._width, y:y+self._width, ...] += \
                util.resizeDensityPatch(np.squeeze(d_predict[i, ...]),
                patch_width)

        # Compensate for overlap
        self._d_buf = np.true_divide(self._d_buf, self._divider)

        # Filter and threshold to eliminate some of the noise.
        self._d_buf = gaussian_filter(self._d_buf, 0.5*self._cfg.DOT_RADIUS_SIGMA)
        self._d_buf[self._d_buf < 0.2*self._d_buf.max()] = 0.0

        # Return prediction.
        return (np.sum(self._d_buf), self._d_buf)

