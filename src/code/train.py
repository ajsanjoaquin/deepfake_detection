# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import datetime
import os
import time

from .architectures import ARCHITECTURES, get_architecture
from .datasets import get_dataset, DATASETS
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from .train_utils import AverageMeter, accuracy, init_logfile, log


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


parser.add_argument('--pretrained-model', type=str, default='',
                    help='Path to a pretrained model')
parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
parser.add_argument('--data_root', type=str, default='train')
parser.add_argument('--test_root', type=str, default='test')
parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, 
        help='the parameter of l2 restriction for weights')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='learning rate')
args = parser.parse_args()


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train', args.data_root, None)
    test_dataset = get_dataset(args.dataset, 'test', None, args.test_root)
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)
    if args.pretrained_model == 'xception_first_time':
        model = get_architecture("xception", args.dataset)
        checkpoint = torch.load(args.load_checkpoint)
        model[1].load_state_dict(checkpoint, strict=False)
    else:
        model = get_architecture(args.arch, args.dataset)

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
    print(len(train_dataset.classes))

    criterion = CrossEntropyLoss().cuda()
    optimizer = Adam(model.parameters(), args.learning_rate, betas=(0.9,0.999), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=[2, 4, 6, 7, 8], 
                                                         gamma=args.gamma)
    best_acc = 0
    best_test_acc = 0
    for epoch in range(args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd)
        scheduler.step(epoch)
        after = time.time()
        print('Train Loss:{}, Train acc:{}, Test Loss:{}, Test acc:{}'.format(train_loss, train_acc,test_loss, test_acc))

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))
        if train_acc >= best_acc and test_acc >= best_test_acc:
            best_acc=train_acc
            best_test_acc=test_acc
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint_%d.pth.tar' % epoch))
    print('Best Train Acc:{}, Best Test Acc:{}'.format(best_acc,best_test_acc))


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    #losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()
    end = time.time()
    correct = 0
    total = 0

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()

        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        '''# measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))'''

        # compute gradient and do Adam step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #metrics
        _, pred = torch.max(outputs.data, dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        std_acc= (correct/total) * 100

        if i % args.print_freq == 0:
            '''print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@2 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))'''
            print('Epoch: [{0}][{1}/{2}]'.format(epoch,i,len(loader)))

    return (loss.item(), std_acc)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()
    end = time.time()

    correct = 0
    total = 0

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            '''acc1, acc5 = accuracy(outputs, targets, topk=(1, 2))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))'''
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(outputs.data, dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            std_acc= (correct/total) * 100

            if i % args.print_freq == 0:
                '''print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@2 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))'''

        return (losses.avg, std_acc)


if __name__ == "__main__":
    main()
