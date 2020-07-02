"""
this code is modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
"""
import sys
sys.path.append("..")

import os
import numpy as np

import torch
import torch.nn as nn

from src.utils import tensor2cuda



def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range; explains why 0 epsilon image is still modified
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def adv_attack(data, label, model, epsilon):
    data=data.clone()
    data.requires_grad = True
    #zero all grads
    model.zero_grad()

    output = model(data)
    criterion=nn.CrossEntropyLoss()

    loss = criterion(output, label)
    #get grads of model in backward pass
    loss.backward()
    #collect data grad
    data_grad=data.grad.data
    #FGSM
    adv_data=fgsm_attack(data, epsilon, data_grad)
    return adv_data

