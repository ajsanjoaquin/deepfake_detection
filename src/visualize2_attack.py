import torch
import torchvision as tv
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
from torchvision import transforms
import matplotlib.pyplot as plt 
import cv2

from src.attack import fgsm_attack
from src.utils import makedirs, tensor2cuda
from src.argument import parser
from .models import model_selection


args=parser()
img_folder = args.output
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################
epsilon=args.epsilon
perturbation_type = 'l2'
label_dict = {0:'fake',1:'real'}

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        for sub_tensor in tensor:
          for t, m, s in zip(sub_tensor, self.mean, self.std):
              t.mul_(s).add_(m)
              # The normalize code -> t.sub_(m).div_(s)
        return tensor



test_transform = transforms.Compose([transforms.Resize((299,299)),
          transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
              ])

unnorm=UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

te_dataset=tv.datasets.ImageFolder(args.data_root,transform=test_transform)
te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

adv_list = []
pred_list = []

#model instantiation
model, *_ = model_selection(modelname='xception', num_out_classes=2)
if device.type=='cpu':
    checkpoint = torch.load(args.load_checkpoint,map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(args.load_checkpoint)
model.load_state_dict(checkpoint)
model.eval()

for data, label in te_loader:
    data, label = tensor2cuda(data), tensor2cuda(label)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    #forward
    output = model(data)
    pred = output.max(1, keepdim=True)[1] #index of max log-probability

    if pred.item()!=label.item():
        continue

    loss= F.nll_loss(output,label)

    #zero all grads
    model.zero_grad()

    #get grads of model in backward pass
    loss.backward()

    #collect data grad
    data_grad=data.grad.data

    #FGSM
    adv_data=fgsm_attack(data,epsilon,data_grad)

    # Re-classify the perturbed image
    adv_pred = model(adv_data)

    final_pred = adv_pred.max(1, keepdim=True)[1] # get predicted label of adversarial data
    if final_pred.item() == label.item():
        correct += 1
#accuracy
final_acc = correct/float(len(te_loader))
print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(te_loader), final_acc))

adv_list.append(adv_data.detach().cpu().numpy().squeeze() * 255.0)  # (N, 28, 28)
pred_list.append(final_pred.cpu().numpy())


data = unnorm(data.detach().cpu()).squeeze().numpy() *255# (N, 28, 28)
label = label.cpu().numpy()

adv_list.insert(0, data)

pred_list.insert(0, label)

if not os.path.isdir(img_folder):
    os.mkdirs(img_folder)

types = ['Original', 'Your Model']

fig, _axs = plt.subplots(nrows=len(adv_list), ncols=len(te_dataset.samples),figsize = [24,7.5])

axs = _axs

for j, _type in enumerate(types):
    axs[j, 0].set_ylabel(_type)
    for i in range(len(te_dataset.samples)):
        #fix predictions label
        axs[j, i].set_xlabel('%s' % label_dict[pred_list[j][i]]) 
        img = adv_list[j][i]
        img = np.transpose(img, (1, 2, 0))

        img = img.astype(np.uint8)
        axs[j, i].imshow(img)

        axs[j, i].get_xaxis().set_ticks([])
        axs[j, i].get_yaxis().set_ticks([])

plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'attack(epsilon: %s)_.jpg' % (args.epsilon)))
print('##done!')