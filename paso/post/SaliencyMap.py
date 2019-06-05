#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
"""
    'based on Deep Inside Convolutional Networks: Visualising Image Classification Models
    and Saliency Maps', https://arxiv.org/abs/1312.6034 
    Also, based loosely on original code by  Nathaniel Jermain
"""

import os
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
from keras.models import load_model
from keras import activations

class  SaliencyMap(object):
    """
        A saliency map  plots the differential (gradient) of the predicted outcome from the 
        model with respect to the input, or pixel values.
        
        `` d(predicted class)/d(pixel value)``
        
        
        In other words. the rate of change of predicted class by change in pixel value;
        By calculating the change in predicted class by applying small delta of change
        to pixel values across the image we can measure the relative importance of each 
        pixel to the ultimate prediction by the model. 

        This technique is described in more detail at 
        https://arxiv.org/abs/1312.6034

    """
    def __init__(self):
        self.image = None
        self.gradient =None
        self.salmap = None
        self.salmap_guassian_smoothed = None

    def plot(self, img, model,layer_name):
        """

            Parameters:
                img
                model (keras model instance) ResNet50
                layer_name (str)

            Return:
                self
        """
        layer = utils.find_layer(model, layer_name)
        model.layers[layer].activation = activations.linear
        model_mod = utils.apply_modifications(model)

        self.salmap = visualize_saliency(model_mod, layer, filter_indices=None,
                                   seed_input= img, backprop_modifier=None, \
                                   grad_modifier="absolute")
        plt.imshow(self.salmap)
#        plt.savefig('SalMapRaw.png', dpi=300)

        self.salmap_guassian_smoothed = ndimage.gaussian_filter(self.salmap[:,:,2], sigma=5)
        plt.imshow(img)
        plt.imshow(self.salmap_guassian_smoothed, alpha=.7)
        plt.axis('off')
        
#        plt.savefig('SalMapScale.png', dpi=300)
        return self