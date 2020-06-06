import torch
import torchvision as tv
import numpy as np
import os, shutil
from torch.utils.data import DataLoader
import sys
from torchvision import transforms
import matplotlib.pyplot as plt 
import cv2
from tqdm import tqdm

from src.utils import makedirs, tensor2cuda
from src.argument import parser
from src.visualization import VanillaBackprop
from src.xception_2 import myxception_
from src.argument import parser

args=parser()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#output location
img_folder = args.output
if not os.path.isdir(img_folder):
    os.mkdir(img_folder)
################################################
test_transform = transforms.Compose([transforms.Resize((299,299)),
          transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
              ])

te_dataset=tv.datasets.ImageFolder(args.data_root,transform=test_transform)
te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

#model instantiation
model = myxception_(num_classes=2, pretrained='imagenet')
if device.type=='cpu':
  checkpoint = torch.load(args.load_checkpoint,map_location=torch.device('cpu'))
else:
  checkpoint = torch.load(args.load_checkpoint)
  model.cuda()
model.load_state_dict(checkpoint['net'])
print('##acc:',checkpoint['acc'])

gradlist=[]

for data, label in tqdm(te_loader):
    data, label = tensor2cuda(data), tensor2cuda(label)

    VBP = VanillaBackprop(model)
    grad = VBP.generate_gradients(data, label)
    grad_flat = grad.view(grad.shape[0], -1)
    mean = grad_flat.mean(1, keepdim=True).unsqueeze(2).unsqueeze(3)
    std = grad_flat.std(1, keepdim=True).unsqueeze(2).unsqueeze(3)

    mean = mean.repeat(1, 1, data.shape[2], data.shape[3])
    std = std.repeat(1, 1, data.shape[2], data.shape[3])

    grad = torch.max(torch.min(grad, mean+3*std), mean-3*std)

    grad -= grad.min()

    grad /= grad.max()

    grad = grad.cpu().numpy().squeeze() *255 # (N, 28, 28)

    gradlist.extend(grad)


types = ['Original', 'Gradient']

#do for each image
for i in tqdm(range(len(te_dataset.samples))):
    img = gradlist[i]
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)

    fig=plt.figure(figsize=[6.4, 4.8])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    try:
        ax.imshow(img)
        plt.savefig(os.path.join(img_folder, 'pair_{}'.format(i+1)),bbox_inches = 'tight', pad_inches = 0, dpi=100)
    except: 
        pass
    #CLEAR FIGURE FOR NEXT IMAGE
    plt.close()

