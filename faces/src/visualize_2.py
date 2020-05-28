import torch
import torchvision as tv
import numpy as np
import os
from torch.utils.data import DataLoader
import sys
from torchvision import transforms
import matplotlib.pyplot as plt 
import cv2

from src.utils import makedirs, tensor2cuda
from src.argument import parser
from src.visualization import VanillaBackprop
from src.xception_2 import myxception_
from src.argument import parser

args=parser()
img_folder = '/content/outputs'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################
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

label_dict = {0:'fake',1:'real'}

for data, label in te_loader:

    data, label = tensor2cuda(data), tensor2cuda(label)


    break

pred_list = []

#model instantiation
with torch.no_grad():
    model = myxception_(num_classes=2, pretrained='imagenet')
    checkpoint = torch.load(args.load_checkpoint)
    print('##acc:',checkpoint['acc'])

    model.load_state_dict(checkpoint['net'])
    model.cuda()
    #get prediction labels
    output = model(data)
    pred = torch.max(output, dim=1)[1]
    pred_list.append(pred.cpu().numpy())

VBP = VanillaBackprop(model)
grad = VBP.generate_gradients(data, label)
grad_flat = grad.view(grad.shape[0], -1)
mean = grad_flat.mean(1, keepdim=True).unsqueeze(2).unsqueeze(3)
std = grad_flat.std(1, keepdim=True).unsqueeze(2).unsqueeze(3)

mean = mean.repeat(1, 1, data.shape[2], data.shape[3])
std = std.repeat(1, 1, data.shape[2], data.shape[3])

grad = torch.max(torch.min(grad, mean+3*std), mean-3*std)

print('##grad:',grad.size())
print(grad.min(), grad.max())

grad -= grad.min()

grad /= grad.max()

grad = grad.cpu().numpy().squeeze() *255 # (N, 28, 28)

label = label.cpu().numpy()

pred_list.insert(0, label)

data = unnorm(data.cpu()).numpy().squeeze() * 255


out_list = [data, grad]

if not os.path.isdir(img_folder):
    os.mkdirs(img_folder)

types = ['Original', 'Your Model']

fig, _axs = plt.subplots(nrows=len(out_list), ncols=len(te_dataset.samples),figsize = [24,7.5])

axs = _axs

for j, _type in enumerate(types):
    axs[j, 0].set_ylabel(_type)
    for i in range(len(te_dataset.samples)):
        #pred_list format: 1st array = true label; 2nd array = predicted label
        axs[j, i].set_xlabel('%s' % label_dict[pred_list[j][i]]) 
        img = out_list[j][i]
        img = np.transpose(img, (1, 2, 0))

        img = img.astype(np.uint8)
        #cv2.imwrite( '_'+str(i)+'_'+str(j)+'.jpg',img)
        axs[j, i].imshow(img)

        axs[j, i].get_xaxis().set_ticks([])
        axs[j, i].get_yaxis().set_ticks([])
        fig.subplots_adjust(hspace=0.025, wspace=0.025)
        plt.tight_layout()
        
plt.savefig(os.path.join(img_folder, '_grad_%s.jpg' % args.affix))
print('##done!')



