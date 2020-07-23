import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision import transforms
from time import time
from .models import model_selection
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

import pandas as pd
from src.argument import parser, print_args
from efficientnet_pytorch import EfficientNet

classes={0:'fake',1:'real'}
class ImageFolderWithPaths(tv.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class Trainer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def train(self, model, tr_loader, va_loader, adv_train=False):
        
        args = self.args
        logger = self.logger

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(0.9,0.999), eps=1e-08, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[2, 4, 6, 7, 8], 
                                                         gamma=0.1)
        acc = 0.0
        valid_acc = 0.0
        best_acc=0
        best_va_acc=0

        correct=0
        total=0

        for epoch in range(1, args.max_epoch+1):
            model.train()
            for data, label, paths in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                
                opt.zero_grad()
                output = model(data)

                loss = criterion(output, label)

                
                loss.backward()
                opt.step()

                _, pred = torch.max(output.data, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                std_acc= (correct/total) * 100

            if va_loader is not None:
                model.eval()
                t1 = time()
                va_acc = self.test(model, va_loader, False, True)
                va_acc = va_acc * 100.0

                t2 = time()
                logger.info('\n'+'='*20 +' evaluation at epoch: %d '%(epoch) \
                    +'='*20)
                logger.info('train acc: %.3f %%, validation acc: %.3f %%, spent: %.3f' % (
                    std_acc, va_acc, t2-t1))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')

            acc = std_acc
            valid_acc = va_acc

            if acc >= best_acc and valid_acc >= best_va_acc:
                best_acc=acc
                best_va_acc=valid_acc
                file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % epoch)
                save_model(model, file_name)
            #for Pytorch 1.0, opt.step() must be called before scheduler.step()
            scheduler.step()
        print('Best Train Acc: {:4f}, Best Valid Acc: {:4f}'.format(best_acc, best_va_acc))
            


    def test(self, model, loader, adv_test=False,valid=False):
        model.eval()
        logger = self.logger
        if valid==False:
            logger.info("Test Set: %d" % len(loader.dataset))
        total = 0
        test_correct=0

        pathlist=[]
        labellist=[]
        predlist=[]

        #turn off backprop (important to avoid cuda memory error)
        with torch.no_grad():
            for data,labels, paths in loader:
                data, labels = tensor2cuda(data), tensor2cuda(labels)
                #forward
                output = model(data)
                #return probabilities for dataframe
                if valid ==False:
                    preds= torch.nn.functional.softmax(output)

                _, pred = torch.max(output.data, 1)

                total += labels.size(0)
                test_correct += (pred == labels).sum().item()
                pathlist.extend(paths)
                labellist.extend(labels)
                if valid==False:
                    predlist.extend(preds)
                

        if valid==False:
            results=pd.DataFrame.from_dict(dict(zip(pathlist,zip(labellist,predlist))),orient='index',columns=['actual','probs']).rename_axis('filename').reset_index()
            results['actual']=results['actual'].apply(lambda x: classes[x.item()])
            results['fake']=results['probs'].apply(lambda x: x[0].item())
            results['real']=results['probs'].apply(lambda x: x[1].item())
            results.drop(['probs'], axis=1, inplace=True)
            #get the column name of the highest probability
            results['predicted'] = results[['fake','real']].idxmax(axis=1)
            results.to_csv(os.path.join(args.log_root,'%s_results.csv'%args.affix))
        
        with open(os.path.join(args.log_root,'%s_out.txt'% args.affix), 'w') as f:
            print('Standard Accuracy: %.4f' % (test_correct / total) ,file=f)
        return test_correct/total

def main(args):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    makedirs(args.log_root)
    makedirs(args.model_folder)

    setattr(args, 'log_folder', args.log_root)
    setattr(args, 'model_folder', args.model_folder)

    logger = create_logger(args.log_root, args.todo, 'info')

    print_args(args, logger)
    model = EfficientNet.from_pretrained('efficientnet-b5', num_classes = 2)
    if args.load_checkpoint is not None:
        if device.type=='cpu':
            checkpoint = torch.load(args.load_checkpoint,map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)
        
    if torch.cuda.device_count() > 1:
        print('GPUs: ', torch.cuda.device_count())
        model = nn.DataParallel(model)
    
    model.to(device)

    trainer = Trainer(args, logger)
    transform = transforms.Compose([transforms.Resize((299,299)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    if args.todo == 'train':
        train_set= ImageFolderWithPaths(args.data_root,transform=transform)
        val_set=ImageFolderWithPaths(args.val_root,transform=transform)
        logger.info('Train Total: %d'%len(train_set))
        logger.info('Val Total: %d'%len(val_set))
        logger.info( "Classes: {}".format(' '.join(map(str, train_set.classes))))

        tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        te_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, adv_train=args.adv)
    elif args.todo == 'test':
        te_dataset=ImageFolderWithPaths(args.data_root,transform=transform)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        std_acc= trainer.test(model, te_loader, adv_test=args.adv)
        print("std acc: %.4f" % (std_acc * 100)

    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)