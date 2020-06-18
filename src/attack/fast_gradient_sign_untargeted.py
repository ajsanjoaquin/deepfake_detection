"""
this code is modified from https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks

original author: Utku Ozbulak - github.com/utkuozbulak
"""
import sys
sys.path.append("..")

import os
import numpy as np

import torch
from torch import nn

from src.utils import tensor2cuda



def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
    
