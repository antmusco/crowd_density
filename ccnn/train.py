#!/usr/bin/env python

"""
Author: Anthony G. Musco
Email:  amusco@cs.stonybrook.edu
Date:   05/01/2017

This file contains the operations to train the counting CNN on generated
features. 

Based off the Hyda CNN model by Daniel Onoro-Rubio and Roberto J. Lopenz
Sastre: https://github.com/gramuah/ccnn
"""


import os
from shutil import copyfile
# Suppress debug prints from TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import ccnn
import util


def main():
    """Train CCNN for a number of epochs."""

    # Load config settings.
    cfg = util.loadConfig()

    # Total patches in the dataset and batch size.
    n_patches = cfg.IMG_NUM * cfg.PATCH_PER_IMG
          
    # Build the training graph.
    train_graph = tf.Graph()
    with train_graph.as_default():

        # Global step.
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Determine the training epoch size.
        train_n_patches = n_patches * cfg.TRAIN_PCT
        train_n_epoch   = int(train_n_patches / cfg.BATCH_SIZE)

        # Create input queue that grabs BATCH_SIZE images at a time.
        train_patches, \
        train_densities = ccnn.inputs('train', 0, cfg)

        # Pass through CCNN, perform optimization.
        train_predict  = ccnn.inference(train_patches, is_train=True, cfg=cfg)
        train_loss     = ccnn.loss(train_predict, train_densities)
        train_optimize = ccnn.optimize(train_loss, global_step, train_n_epoch, cfg)

        # Log the images in TensorBoard.
        tf.summary.image('patches', train_patches)
        tf.summary.image('predict', train_predict)
        tf.summary.image('truth',   train_densities)

        # Restore the last model and train for a number of epochs.
        train_saver = tf.train.Saver()
        last_step   = train_n_epoch * cfg.CNN_NUM_EPOCHS

        # Use a monitored training session to automatically save checkpoints
        # and summaries during training. We also use a custom logger,
        # 'ccnn.LoggerHook', to monitor the progress from the terminal.
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=cfg.TRAIN_CHECKPOINT_DIR,
            hooks=[
                    tf.train.StopAtStepHook(last_step=last_step),
                    tf.train.NanTensorHook(train_loss), 
                    ccnn.LoggerHook('Train loss', train_loss, cfg)
                ],
            config=tf.ConfigProto(log_device_placement=cfg.LOG_DEVICE_PLACE),
            save_checkpoint_secs=120,
            save_summaries_steps=100) as train_sess:

            # Load the last checkpoint, initialize the global step.
            ckpt = tf.train.get_checkpoint_state(cfg.TRAIN_CHECKPOINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                train_saver.restore(train_sess, ckpt.model_checkpoint_path)

            # Run the training session.
            step  = 0
            epoch = 1
            while not train_sess.should_stop():
                if step % train_n_epoch == 0:
                    # Output checkpoint as model:
                    copyfile(os.path.join(cfg.TRAINING_CHECKPOINT_DIR,
                        'checkpoint'), cfg.CNN_MODEL_OUTPUT)
                    print '------- EPOCH {} -------'.format(epoch)
                    print 'Running {} batches ({} patches with batch size {})'.format(
                          train_n_epoch, train_n_patches, cfg.BATCH_SIZE)
                    epoch += 1
                step += 1 
                train_sess.run(train_optimize)


# Entry point.
if __name__ == '__main__':
    main()

