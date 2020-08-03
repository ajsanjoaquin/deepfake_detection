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
        os.makedirs(folder)
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
global_max = 0
global_min = 0

if args.normalize == 'global' or args.normalize == 'local':
  for data, label in tqdm(te_loader):
      data, label = tensor2cuda(data), tensor2cuda(label)
      tf=np.ones(args.batch_size, dtype=np.int64)
      x=torch.from_numpy(tf)
      x=x.cuda()

      VBP = VanillaBackprop(model)
      grad = VBP.generate_gradients(data, x)
      
      grad = grad.cpu().numpy().squeeze() # (batch_size, channels, 288, 288)
      if args.normalize == 'local':
        '''
        get local extremes for each image array 
        then normalize image array with their corresponding local extremes
        '''
        max=np.amax(grad, axis = (1,2,3))
        min=np.amin(grad, axis = (1,2,3))
        max=max.reshape([args.batch_size,1,1,1])
        min=min.reshape([args.batch_size,1,1,1])
        normalized = np.divide((np.subtract(grad, min)), (max - min))
        labellist.extend(label)
        gradlist.extend(normalized)

        labellist.extend(label)
        gradlist.extend(normalized)

      elif args.normalize == 'global':
        max=np.amax(grad)
        min=np.amin(grad)
        if max > global_max:
          global_max = max
        if min > global_min:
          global_min = min

elif args.normalize == 'sign':
  for data, label in tqdm(te_loader):
      data, label = tensor2cuda(data), tensor2cuda(label)
      tf=np.ones(args.batch_size, dtype=np.int64)
      x=torch.from_numpy(tf)
      x=x.cuda()

      VBP = VanillaBackprop(model)
      grad = VBP.generate_gradients(data, x)
      
      grad = grad.cpu().numpy().squeeze()
      normalized=np.sign(grad) * args.epsilon
      labellist.extend(label)
      gradlist.extend(normalized)

else: raise NotImplementedError

#second pass on data to normalize it after finding global extremes
if args.normalize == 'global':
  for data, label in tqdm(te_loader):

    data, label = tensor2cuda(data), tensor2cuda(label)
    tf=np.ones(args.batch_size, dtype=np.int64)
    x=torch.from_numpy(tf)
    x=x.cuda()

    VBP = VanillaBackprop(model)
    grad = VBP.generate_gradients(data, x)
    grad = grad.cpu().numpy().squeeze()

    normalized = (grad - global_min) / (global_max - global_min)
    labellist.extend(label)
    gradlist.extend(normalized)

# saving as np array
for i in tqdm(range(len(te_dataset.samples))):
    if label_dict[labellist[i].item()]=='fake':
        np.save(os.path.join(fake, '{}_{}'.format(basename(args.output),i+1)), gradlist[i])
    else:
        np.save(os.path.join(real, '{}_{}'.format(basename(args.output),i+1)), gradlist[i])
