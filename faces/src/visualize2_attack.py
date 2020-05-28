import torch
import torchvision as tv
import numpy as np
import os
from torch.utils.data import DataLoader
import sys
from torchvision import transforms
import matplotlib.pyplot as plt 
import cv2

from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, tensor2cuda
from src.argument import parser
from src.xception_2 import myxception_


args=parser()
img_folder = '/content/outputs'
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################
max_epsilon = 4.7
perturbation_type = 'l2'
label_dict = {0:'fake',1:'real'}

test_transform = transforms.Compose([transforms.Resize((299,299)),
          transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
              ])
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

unnorm=UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

te_dataset=tv.datasets.ImageFolder(args.data_root,transform=test_transform)
te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


for data, label in te_loader:

    data, label = tensor2cuda(data), tensor2cuda(label)


    break

adv_list = []
pred_list = []

#model instantiation
with torch.no_grad():
    model = myxception_(num_classes=2, pretrained='imagenet')
    checkpoint=torch.load(args.load_checkpoint)
    print('##acc:',checkpoint['acc'])
    model.load_state_dict(checkpoint['net'])
    model.cuda()

    attack = FastGradientSignUntargeted(model, 
                                        max_epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=perturbation_type)

   #references of _eval=true must be removed
    adv_data = attack.perturb(data, label, 'mean', False)

    output = model(adv_data)
    pred = torch.max(output, dim=1)[1]
    adv_list.append(adv_data.cpu().numpy().squeeze() * 255.0)  # (N, 28, 28)
    pred_list.append(pred.cpu().numpy())


data = unnorm(data.cpu()).squeeze().numpy() *255# (N, 28, 28)
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
plt.savefig(os.path.join(img_folder, 'attack_%s_%s.jpg' % (perturbation_type,args.affix)))
print('##done!')