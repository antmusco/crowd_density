#!/usr/bin/env python

"""
Author: Anthony G. Musco
Email:  amusco@cs.stonybrook.edu
Date:   05/01/2017

Script used to test the CCNN.
"""


import os
# Suppress debug prints from TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from   mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
import ccnn
import util


def dispColorHistogram(miscount, path):
    """ Displays a color histogram based on the miscount matrix. """
    fig, ax = plt.subplots()
    x, y    = miscount[:,0], miscount[:,2]
    hist    = ax.hist2d(x, y, bins=10, norm=LogNorm())
    fig.colorbar(hist[3], ax=ax, cmap='BuGn')
    ax.set_xlabel('True Count')
    ax.set_ylabel('Miscount')
    fig.suptitle('Miscount vs. True Count Histogram')
    print "Saving figure..."
    fig.savefig(path)
    print "done."


def main():
    """ Test CCNN. """

    # Load the config settings.
    cfg = util.loadConfig()

    # Total patches in the dataset.
    n_patches  = cfg.IMG_NUM * cfg.PATCH_PER_IMG

    # Build the test graph.
    test_graph = tf.Graph()
    with test_graph.as_default():

        # Determine the testing size.
        test_n_patches = n_patches * cfg.TEST_PCT
        test_n_epoch   = int(test_n_patches / cfg.BATCH_SIZE)

        # Pass through CCNN>
        test_patches, \
        test_densities = ccnn.inputs('test', 0, cfg)
        test_predict   = ccnn.inference(test_patches, is_train=True, cfg=cfg)
        test_loss      = ccnn.miscount(test_predict, test_densities, totals=True)

        # Train on the entire training set.
        test_saver = tf.train.Saver()
        with tf.Session() as test_sess:

            # Load the last training checkpoint.
            ckpt = tf.train.get_checkpoint_state(cfg.TRAIN_CHECKPOINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                test_saver.restore(test_sess, ckpt.model_checkpoint_path)
            else:
                print 'No checkpoint found!'

            # SUPER IMPORTANT.
            # Start filename queue runner.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            print 'Running {} batches ({} patches with batch size {})'.format(
                  test_n_epoch, test_n_patches, cfg.BATCH_SIZE)

            # Run the testing session.
            num = test_n_epoch
            miscount = np.zeros([num * cfg.BATCH_SIZE, 3])
            for step in xrange(num):
                # Slice for batch totals.
                s = slice(step*cfg.BATCH_SIZE, (step+1)*cfg.BATCH_SIZE)
                miscount[s, 0], miscount[s, 1], miscount[s, 2] = \
                    np.squeeze(test_sess.run(test_loss))
                print 'Batch: {} -- True count: {:05.2f} Predict count: {:05.2f}'\
                    ' (err {:04.2f} - {:05.2f}%)'.format(
                        step, 
                        np.mean(miscount[s, 0]),
                        np.mean(miscount[s, 1]), 
                        np.mean(miscount[s, 2]),
                        (100. * np.mean(miscount[s, 2])/np.mean(miscount[s, 0]))
                    )

            # Finish off filename queue coordinator.
            coord.request_stop()
            coord.join(threads)

            # Display histogram.        
            path = os.path.join(cfg.TEST_CHECKPOINT_DIR, 'hist.png')
            dispColorHistogram(miscount, path)


# Entry point.
if __name__ == '__main__':
    main()

