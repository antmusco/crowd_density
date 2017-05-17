#!/usr/bin/env python

"""
Author: Anthony G. Musco
Email:  amusco@cs.stonybrook.edu
Date:   05/01/2017

This file contains the logic convert raw images and their 'dotted' labels to
tensorflow record files for CCNN input.
"""

import os
import tensorflow as tf
TFRecordWriter = tf.python_io.TFRecordWriter
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from   scipy.ndimage.filters import gaussian_filter 
from   skimage.transform import resize

# Counting CNN
import ccnn


def main():
    """ Script which generates the example sets from the image sets. """
    # Load config file.
    print "Loading configuration file..."
    cfg = ccnn.util.loadConfig()
    # Initialize patch scale writer.
    with PatchScaleWriter(cfg) as writer:
        # Generate training set:
        for img_set in ['train', 'valid', 'test']:
            _genExamples(img_set, writer, cfg)


class PatchScaleWriter(object):
    """
    Class which writes a patch and it's corresponding denisty map as an example
    at several scales. The class writes only a limited number of examples per
    file, in order to keep each file small and manageable. It does this by
    maintaining a list of open TFRecordWriters for writing examples to "chunk"
    files for TensorFlow to read using the Reader Operations.
    """
    def __init__(self, cfg):
        """
        Constructor.
        Args:
            cfg: Configuration parameters.
        """
        self._num_scales = cfg.CNN_N_SCALES
        self._chunk_num  = 0
        self._chunk_max  = cfg.BATCH_SIZE * cfg.BATCH_PER_CHUNK
        self._writers    = {}
        self._set_lists  = {}
        self._set_size   = {}
        self._chunk_fmt  = os.path.join(cfg.EXL_DIR, cfg.CHUNK_FMT)
        self._example_set_fmt = os.path.join(cfg.EXL_SET_DIR, cfg.EXL_SET_FMT)
        self._display    = cfg.DEBUG_DISPLAY

    def __enter__(self):
        # Nothing to do.
        return self

    def __exit__(self, *args):
        # Close all open writers.
        for example_set in self._writers:
            for s in xrange(self._num_scales):
                self._writers[example_set][s].close()
                self._set_lists[example_set][s].close()

    def _updateWriter(self, example_set):
        """ 
        When an example set reaches its maximum size, this function closes the
        writers at each scale for the indicated example set, and opens new
        writers for a new chunk. It also records the new chunk in the chunk
        list for this example set.
        Args:
            example_set: Either 'train', 'valid', or 'test'.
        """
        # If this example set has not been initialized, initialize it.
        if example_set not in self._writers:
            self._writers[example_set] = [None] * self._num_scales
            self._set_lists[example_set] = { s: \
                open((self._example_set_fmt % (example_set, s)), 'a') \
                for s in xrange(self._num_scales) }
        # Open a new chunk file for each scale, and create a new writer for
        # each new chunk file.
        for s in xrange(self._num_scales):
            # Generate name for new chunk file.
            chunk_file = self._chunk_fmt % (s, self._chunk_num)
            # Close current writer and create new writer for chunk file.
            if self._writers[example_set][s] is not None:
                self._writers[example_set][s].close()
            self._writers[example_set][s] = TFRecordWriter(chunk_file)
            # Record newly create chunk file in example set list.
            self._set_lists[example_set][s].write(chunk_file + '\n')
            # Reset example set size.
            self._set_size[example_set] = 0
        # On to the next chunk.
        self._chunk_num += 1

    def _scalePatch(self, patch, s):
        """ 
        Crop a patch to the desired scale and resize it so that it's the same
        size as the original patch. 
        Args:
            patch: Patch to scale of size [H, W, C]
            s: Scale to output, 0 <= s <= self._num_scales
        Returns:
            The cropped and rescaled image patch.
        """
        h, w    = patch.shape[:2]
        scale_h = (s * h) / (2 * self._num_scales)
        scale_w = (s * w) / (2 * self._num_scales)
        patch_s = patch[scale_h : h - scale_h, scale_w : w - scale_w]
        # Make sure every pixel is in range [-1.0, 1.0]
        patch_s[patch_s >  1.0] =  1.0
        patch_s[patch_s < -1.0] = -1.0
        patch_s = resize(patch_s, (h, w))
        return patch_s.astype(np.float32)

    def writeScales(self, patch, density, example_set):
        """
        Writes a patch/density pair at different scales for the indicated
        example set.
        Args:
            patch: The image patch, resized to CNN_INPUT_WIDTH
            density: The density map, resized to CNN_OUTPUT_WIDTH
            example_set: String name of the example set to write to.
        """
        # Debug display to make sure everything is working.
        if self._display:
            dens = resize(density, patch.shape[:2])
            fig, ax = plt.subplots()
            im = ax.pcolormesh(dens, alpha=0.5)
            fig.colorbar(im, ax=ax)
            ax.imshow(patch, aspect='auto')
            plt.show()
        # Initialize example set if needed.
        if example_set not in self._writers:
            self._updateWriter(example_set)
        # Write each scale.
        for s in xrange(self._num_scales):
            # Scale source patch.
            patch_s   = self._scalePatch(patch, s)
            density_s = self._scalePatch(density, s)
            example   = ccnn.ccnn.serializeExample(patch_s, density_s)
            self._writers[example_set][s].write(example.SerializeToString())
        # Check to see if we reached the chunk size limit for this example set.
        # If so, update the writers.
        self._set_size[example_set] += 1
        if self._set_size[example_set] >= self._chunk_max:
            self._updateWriter(example_set)


def _genExamples(img_set, writer, cfg):
    """ 
    Generates examples using images from the image set.
    
    Each image will be sampled a number of times to generate patches used for
    the example set. If we are training at different sales, then each sampled
    patch will be cropped several times to generate more examples at smaller
    and smaller scales. Each of these cropped images will be resized to the
    original size, so that all scales can be passed into their respective CCNN.

    Args:
        img_set: Either 'train', 'valid', or 'test'.
        writer: An open instance of PatchScaleWriter used for writing patches.
        cfg: Configuration parameters.
    """
    print "Generating '{}' example set...".format(img_set)
    count = 0

    # Extract config settings to variables for brevity.
    patch_width = cfg.PATCH_WIDTH
    num_patches = cfg.PATCH_PER_IMG
    input_size  = (cfg.CNN_INPUT_WIDTH, cfg.CNN_INPUT_WIDTH)
    output_size = (cfg.CNN_OUTPUT_WIDTH, cfg.CNN_OUTPUT_WIDTH)

    # For each image extract a number of patches from the source image and the
    # dot label image.
    print "Reading image file names..."
    img_set_path = os.path.join(cfg.IMG_SET_DIR, (cfg.IMG_SET_FMT % img_set))
    for img_file in np.loadtxt(img_set_path, dtype='str'):

        # Get image paths, read image, generate density image.
        print "Processing image: {}".format(img_file)
        img_file = os.path.join(cfg.IMG_DIR, img_file)
        dot_file = img_file.replace(cfg.IMG_EXT, cfg.DOT_EXT)
        img      = mpimg.imread(img_file)
        dot      = mpimg.imread(dot_file)
        dens     = gaussian_filter(dot, cfg.DOT_RADIUS_SIGMA)

        # Randomly sample a number of patch positions at random locations, and
        # collect source and density patches at each position.
        for pos in _samplePatchPositions(img.shape, patch_width, num_patches):

            # Grab patches and resize them.
            patch   = _getPatchAtPosition(img, pos, patch_width)
            density = _getPatchAtPosition(dens, pos, patch_width)
            patch   = resize(patch, input_size).astype(np.float32)
            density = ccnn.util.resizeDensityPatch(density,
                        output_size).astype(np.float32)

            # Write patch/density pair at different scales.
            writer.writeScales(patch, density, img_set)
            count += 1

            # If flips are included, repeat process with flipped image.
            if cfg.INCLUDE_FLIPS:
                writer.writeScales(np.fliplr(patch), np.fliplr(density),
                    img_set)
                count += 1

    print "Generated {} examples".format(count)
    print "--------------------"    
    print "Finished!"


def _samplePatchPositions(img_size, patch_width, N):
    """
    Generates N random positions such that patch_width still fits within the
    image bounds.
    Args:
        img_size: List of ints (height, width).
        patch_width: Square side of the patches as an int.
        N: Number of positions to generate as an int.
    Returns:
        Numpy array of N pixel coordinates for patches.
    """
    h,w = img_size[:2]
    dx = dy = patch_width/2
    y  = np.random.randint(dy, w-dy, N).reshape(N,1)
    x  = np.random.randint(dx, h-dx, N).reshape(N,1)
    return np.hstack((x,y))


def _getPatchAtPosition(img, pos, patch_width):
    """
    Crop patch from img at the position pos with a with of 'patch_width'.
    Args:
        img: input image.
        pos: position to crop at.
        patch_width: Square side of the patches as an int.
    Returns: 
        Cropped square patch with a side of 'patch_width'.
    """
    dx = dy = patch_width/2
    x, y = pos
    sx = slice(x-dx, x+dx+1, None)
    sy = slice(y-dy, y+dy+1, None)
    return img[sx,sy,...]


# Entry point.
if __name__ == "__main__":
    main()
