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
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#output location
img_folder = args.output
tp=os.path.join(args.output,'true_positives')
tn=os.path.join(args.output,'true_negatives')
fp=os.path.join(args.output,'false_positives')
fn=os.path.join(args.output,'false_negatives')
locations=[img_folder,tp,tn,fp,fn]

for folder in locations:
    if not os.path.isdir(folder):
        mkdir(folder)
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
    if device.type=='cpu':
      checkpoint = torch.load(args.load_checkpoint,map_location=torch.device('cpu'))
    else:
      checkpoint = torch.load(args.load_checkpoint)
      model.cuda()
    model.load_state_dict(checkpoint['net'])
    print('##acc:',checkpoint['acc'])
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

grad -= grad.min()

grad /= grad.max()

grad = grad.cpu().numpy().squeeze() *255 # (N, 28, 28)

label = label.cpu().numpy()

pred_list.insert(0, label)

data = unnorm(data.cpu()).numpy().squeeze() * 255


out_list = [data, grad]

types = ['Original', 'Gradient']

#do for each image
for j in range(len(te_dataset.samples)):
    fig, axs = plt.subplots(1, ncols=len(out_list),figsize = [15,15])
    #make gradient for each image
    for i, _type in enumerate(types):
        axs[i].set_ylabel(_type)
        #pred_list format: 1st array = true label; 2nd array = predicted label
        axs[i].set_xlabel('%s' % label_dict[pred_list[i][j]]) 
        img = out_list[i][j]
        img = np.transpose(img, (1, 2, 0))

        img = img.astype(np.uint8)
        axs[i].imshow(img)

        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])
        fig.subplots_adjust(hspace=0.025, wspace=0.025)
        plt.tight_layout()

    plt.savefig(os.path.join(img_folder, 'pair_{}'.format(j+1)))
    # Save the gradient as individual image
    extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(img_folder,'gradient_{}'.format(j+1)), bbox_inches=extent)

    #moving code
    #DEFINE: True Negative = Real images classified as real
    if (axs[0].xaxis.get_label_text()=='real' and axs[0].xaxis.get_label_text()==axs[1].xaxis.get_label_text()):
        shutil.move(os.path.join(img_folder,'pair_{}.png'.format(j+1)),os.path.join(tn,'pair_{}.png'.format(j+1)))
        shutil.move(os.path.join(img_folder,'gradient_{}.png'.format(j+1)),os.path.join(tn,'gradient_{}.png'.format(j+1)))
    if (axs[0].xaxis.get_label_text()=='fake' and axs[0].xaxis.get_label_text()==axs[1].xaxis.get_label_text()):
        shutil.move(os.path.join(img_folder,'pair_{}.png'.format(j+1)),os.path.join(tp,'pair_{}.png'.format(j+1)))
        shutil.move(os.path.join(img_folder,'gradient_{}.png'.format(j+1)),os.path.join(tp,'gradient_{}.png'.format(j+1)))
    #False Positive
    if (axs[0].xaxis.get_label_text()=='real' and axs[0].xaxis.get_label_text()!=axs[1].xaxis.get_label_text()):
        shutil.move(os.path.join(img_folder,'pair_{}.png'.format(j+1)),os.path.join(fp,'pair_{}.png'.format(j+1)))
        shutil.move(os.path.join(img_folder,'gradient_{}.png'.format(j+1)),os.path.join(fp,'gradient_{}.png'.format(j+1)))
    #False Negative
    if (axs[0].xaxis.get_label_text()=='fake' and axs[0].xaxis.get_label_text()!=axs[1].xaxis.get_label_text()):
        shutil.move(os.path.join(img_folder,'pair_{}.png'.format(j+1)),os.path.join(fn,'pair_{}.png'.format(j+1)))
        shutil.move(os.path.join(img_folder,'gradient_{}.png'.format(j+1)),os.path.join(fn,'gradient_{}.png'.format(j+1)))
    #CLEAR FIGURE FOR NEXT IMAGE
    plt.clf()
print('##done!')



