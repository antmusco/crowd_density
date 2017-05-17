import os
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
from   skimage.transform import resize
from   easydict import EasyDict as edict


# Default configuration file location.
CONFIG_FILE = 'config/ccnn_cfg.yml'


def loadConfig():
    """Load configuration from a file."""
    with open(CONFIG_FILE, 'r') as f:
      yaml_cfg = edict(yaml.load(f))
    return yaml_cfg


def resizeDensityPatch(patch, output_size):
    """
    Take a density map and resize it to the output_size.
    Args:
        patch: input density map.
        output_size: output size.
    Returns: 
        Resized version of the density map.    
    """
    # Get patch size
    h, w = patch.shape[:2]
    # Total sum
    patch_sum = patch.sum()
    # Normalize values between 0 and 1.
    p_max = patch.max()
    p_min = patch.min()
    # Avoid 0 division
    if patch_sum != 0:
        patch = (patch - p_min)/(p_max - p_min)
    # Resize
    patch = resize(patch, output_size)
    # Return back to the previous scale
    patch = patch*(p_max - p_min) + p_min
    # Keep count
    res_sum = patch.sum()
    if res_sum != 0:
        return patch * (patch_sum/res_sum)
    return patch


def densityMapOverlay(patch, dens, fig=None, ax=None):
    # Resize density patch.
    shape = patch.shape[:2]
    if patch != dens.shape[:2]:
        dens = resizeDensityPatch(dens, shape)
    # Plot the density map.
    if fig is None:
        fig, ax = plt.subplots()
    cmesh   = ax.pcolormesh(dens, alpha=0.4, linewidth=0, rasterized=True)
    #fig.colorbar(cmesh, ax=ax)
    # Display the image/density map overlay.
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(patch, extent=[0, shape[0], shape[1], 0])
    return (fig, ax)

