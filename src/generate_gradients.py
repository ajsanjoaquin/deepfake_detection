import torch
import torchvision as tv
import numpy as np
import os, shutil
from os.path import basename
from torch.utils.data import DataLoader
import sys
from torchvision import transforms
import cv2
from tqdm import tqdm

from src.utils import makedirs, tensor2cuda
from src.argument import parser
from src.visualization import VanillaBackprop
from .models import model_selection
from src.argument import parser

args=parser()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#output location
img_folder = args.output
fake=os.path.join(img_folder,'fake')
real=os.path.join(img_folder,'real')
folders=[img_folder,fake,real]
for folder in folders:
    if not os.path.isdir(folder):
        os.mkdir(folder)
################################################
test_transform = transforms.Compose([transforms.Resize((299,299)),
          transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
              ])

te_dataset=tv.datasets.ImageFolder(args.data_root,transform=test_transform)
te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

#model instantiation
model, *_ = model_selection(modelname='xception', num_out_classes=2)
if device.type=='cpu':
  checkpoint = torch.load(args.load_checkpoint,map_location=torch.device('cpu'))
else:
  checkpoint = torch.load(args.load_checkpoint)
  model.cuda()
model.load_state_dict(checkpoint)

gradlist=[]
labellist=[]
label_dict = {0:'fake',1:'real'}


for data, label in tqdm(te_loader):
    data, label = tensor2cuda(data), tensor2cuda(label)
    #make every target label real
    tf=np.ones(len(data), dtype=np.int64)
    x=torch.from_numpy(tf)
    x=x.cuda()

    VBP = VanillaBackprop(model)
    grad = VBP.generate_gradients(data, x)
    
    #normalization
    '''grad_flat = grad.view(grad.shape[0], -1)
    mean = grad_flat.mean(1, keepdim=True).unsqueeze(2).unsqueeze(3)
    std = grad_flat.std(1, keepdim=True).unsqueeze(2).unsqueeze(3)

    mean = mean.repeat(1, 1, data.shape[2], data.shape[3])
    std = std.repeat(1, 1, data.shape[2], data.shape[3])

    grad = torch.max(torch.min(grad, mean+3*std), mean-3*std)
    grad -= grad.min()
    grad /= grad.max()'''
    #unnormalized
    grad = grad.cpu().numpy().squeeze() *255 # (N, 28, 28)

    labellist.extend(label)
    gradlist.extend(grad)

#do for each image
for i in tqdm(range(len(te_dataset.samples))):
    img = gradlist[i]
    img = np.transpose(img, (1, 2, 0))
    
    #for normalized only
    #img = img.astype(np.uint8)

    try:
        if label_dict[labellist[i].item()]=='fake':
            cv2.imwrite(os.path.join(fake, '{}_grad_{}.png'.format(basename(args.output),i+1)),img)
        else:
            cv2.imwrite(os.path.join(real, '{}_grad_{}.png'.format(basename(args.output),i+1)),img)
    except: 
        pass

