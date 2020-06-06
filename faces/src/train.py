import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision import transforms
from time import time
from src.xception_2 import myxception_
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from src.argument import parser, print_args

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger
        #child=model.children()[0]
        #for param in child.parameters():
        #param.requires_grad = False

        opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[2, 4, 6, 7, 8], 
                                                         gamma=0.1)
        _iter = 0

        begin_time = time()
        best_acc = 0.0

        for epoch in range(1, args.max_epoch+1):
            scheduler.step()
            model.train()
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)
                #normalize loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:
                    t1 = time()

                    if adv_train:
                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            adv_output = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:
                        with torch.no_grad():
                            stand_output = model(data, _eval=True)
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                        

                    t2 = time()

                    print('%.3f' % (t2 - t1))

                    logger.info('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (
                        epoch, _iter, time()-begin_time, loss.item()))

                    logger.info('standard acc: %.3f %%, robustness acc: %.3f %%' % (
                        std_acc, adv_acc))

                    # begin_time = time()

                    # if va_loader is not None:
                    #     va_acc, va_adv_acc = self.test(model, va_loader, True)
                    #     va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    #     logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                    #     logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    #         va_acc, va_adv_acc, time() - begin_time))
                    #     logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                _iter += 1

            if va_loader is not None:
                model.eval()
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n'+'='*20 +' evaluation at epoch: %d iteration: %d '%(epoch, _iter) \
                    +'='*20)
                logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    va_acc, va_adv_acc, t2-t1))
                logger.info('='*28+' end of evaluation '+'='*28+'\n')
            if std_acc>best_acc:
                best_acc=std_acc
                file_name = os.path.join(args.model_folder, 'checkpoint_%d.pth' % epoch)
                save_model(model, file_name)
        
        print('Best val Acc: {:4f}'.format(best_acc))
            


    def test(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', 
                                                       False)

                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num

def main(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_folder = args.log_root

    makedirs(log_folder)
    makedirs(args.model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', args.model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    model = myxception_(num_classes=2, pretrained='imagenet')
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['net'])

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        transform = transforms.Compose([transforms.Resize((299,299)),
                transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
        tr_dataset=tv.datasets.ImageFolder(args.data_root,transform=transform)
        #split 80% train, 20% val
        train_set, val_set = torch.utils.data.random_split(tr_dataset,[(len(tr_dataset)*0.80),(len(tr_dataset)*0.20)])

        tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        te_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        te_dataset=tv.datasets.ImageFolder(args.data_root,transform=transform)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)

        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))

    else:
        raise NotImplementedError
    



if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)