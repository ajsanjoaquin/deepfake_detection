import argparse

def parser():
    parser = argparse.ArgumentParser(description='Arguments for All scripts')
    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
        help='what behavior want to do: train | valid | test | visualize')
    parser.add_argument('--model', choices=['xception', 'enet'], help='what model to use')
    parser.add_argument('--data_root', default='/content/train', 
        help='train or test dataset root')
    parser.add_argument('--val_root', help='validation dataset root')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_folder', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--affix', default='', help='the affix for the save folder')
    parser.add_argument('--output', help='specify location of output')

    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--nworkers', type=int, default=4, help='number of workers')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=200, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, 
        help='the parameter of l2 restriction for weights')

    parser.add_argument('--adv', action='store_true', help='set to adversarial training')
    parser.add_argument('--epsilon', '-e', type=float, default=0.0157, help='maximum perturbation of adversaries (4/255=0.0157)')
    parser.add_argument('--init_load', default= None , help= 'path to imagenet weights if finetuning model from imagenet checkpoint')
    parser.add_argument('--normalize', choices=['local','global','sign'],default = None, help = 'For generating gradients script; type of normalization')
    parser.add_argument('--array', default= None, help= 'set to True to when using numpy arrays for training and testing model')
    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))