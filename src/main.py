import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision import transforms
from time import time
from .models import model_selection
#from src.xception_2 import myxception_
from src.attack import adv_attack
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

import pandas as pd
from src.argument import parser, print_args
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


    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader, adv_train=False):
        
        args = self.args
        logger = self.logger
        #child=model.children()[0]
        #for param in child.parameters():
        #param.requires_grad = False
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(0.9,0.999), eps=1e-08, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[2, 4, 6, 7, 8], 
                                                         gamma=0.1)
        acc = 0.0
        valid_acc = 0.0

        logger.info("Train: %d, Validation: %d" % (len(tr_loader.dataset),len(va_loader.dataset)))
        for epoch in range(1, args.max_epoch+1):
            model.train()
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                
                opt.zero_grad()
                output = model(data)
                if adv_train:
                    output = adv_attack (data, label, model, args.epsilon)

                #normalize loss
                loss = criterion(output, label)
                #loss=loss/loss.detach()

                
                loss.backward()
                opt.step()
                
                if adv_train:
                    _, adv_pred = torch.max(output.data, dim=1)
                    correct += (adv_pred == label).sum()
                    total += label.size(0)

                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy()) * 100
                    std_acc = -1


                else:
                    _, pred = torch.max(output.data, dim=1)
                    correct += (pred == label).sum()
                    total += label.size(0)
                    std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

            if va_loader is not None:
                model.eval()
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, False, True)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n'+'='*20 +' evaluation at epoch: %d '%(epoch) \
                    +'='*20)
                if adv_train: logger.info('robust acc: %.3f %%, robust validation acc: %.3f %%, spent: %.3f' % (
                    adv_acc, va_adv_acc, t2-t1))
                else: logger.info('train acc: %.3f %%, validation acc: %.3f %%, spent: %.3f' % (
                    std_acc, va_acc, t2-t1))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')

            if adv_train:
                acc = adv_acc
                valid_acc = va_adv_acc
            else:
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
        # adv_test is False, return adv_acc as -1 
        model.eval()
        logger = self.logger
        if valid==False:
            logger.info("Test Set: %d" % len(loader.dataset))
        adv_correct = 0
        total = 0
        test_correct=0

        pathlist=[]
        labellist=[]
        predlist=[]
        
        for data,labels, paths in loader:
            data, labels = tensor2cuda(data), tensor2cuda(labels)
            #forward
            output = model(data)
            #return probabilities for dataframe
            preds= torch.nn.functional.softmax(output)

            _, pred = torch.max(output.data, 1)

            total += labels.size(0)
            test_correct += (pred == labels).sum().item()


            if adv_test:
                #if already incorrect, don't attack it anymore
                #fix for more than 1 batch size
                if (pred !=labels).item():
                    continue

                # Re-classify the perturbed image
                adv_data = adv_attack (data, labels, model, args.epsilon)
                adv_out = model(adv_data)
                preds= torch.nn.functional.softmax(adv_out)
                _, adv_pred = torch.max(adv_out.data , dim=1)

                total += labels.size(0)
                adv_correct += (adv_pred == labels).sum().item()
            else:
                adv_correct = -total

            pathlist.extend(paths)
            labellist.extend(labels)
            predlist.extend(preds)
        results=pd.DataFrame.from_dict(dict(zip(pathlist,zip(labellist,predlist))),orient='index',columns=['actual','probs']).rename_axis('filename').reset_index()
        results['actual']=results['actual'].apply(lambda x: classes[x.item()])
        results['fake']=results['probs'].apply(lambda x: x[0].item())
        results['real']=results['probs'].apply(lambda x: x[1].item())
        results.drop(['probs'], axis=1, inplace=True)
        #get the column name of the highest probability
        results['predicted'] = results[['fake','real']].idxmax(axis=1)
        results.to_csv('%s_results.csv'%args.affix)
        
        with open('%s_out.txt'% args.affix, 'w') as f:
            print('Standard Accuracy: %.4f, Adversarial Accuracy: %.4f' % (test_correct / total, adv_correct / total) ,file=f)
        return test_correct/total , adv_correct / total

def main(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_folder = args.log_root

    makedirs(log_folder)
    makedirs(args.model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', args.model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    #model = myxception_(num_classes=2, pretrained='imagenet')
    if device.type=='cpu':
        checkpoint = torch.load(args.load_checkpoint,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger)
    transform = transforms.Compose([transforms.Resize((299,299)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    if args.todo == 'train':
        tr_dataset=tv.datasets.ImageFolder(args.data_root,transform=transform)
        logger.info('Total: %d'%len(tr_dataset))
        logger.info( "Classes: {}".format(' '.join(map(str, tr_dataset.classes))))
        #split 80% train, 20% val
        train_set, val_set = torch.utils.data.random_split(tr_dataset,[round(len(tr_dataset)*0.80),round(len(tr_dataset)*0.20)])

        tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        te_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, adv_train=args.adv)
    elif args.todo == 'test':
        te_dataset=ImageFolderWithPaths(args.data_root,transform=transform)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=args.adv)
        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))

    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)