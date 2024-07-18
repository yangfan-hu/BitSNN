import argparse
# import os
"""
args
"""

parser = argparse.ArgumentParser(description='ReCU')

parser.add_argument(
    '--results_dir',
    metavar='RESULTS_DIR',
    default='./results',
    help='results dir')

parser.add_argument(
    '--save',
    metavar='SAVE',
    default='',
    help='saved folder (named by datetime)')

parser.add_argument(
    '--resume',
    dest='resume',
    action='store_true',
    help='resume to the latest checkpoint')

parser.add_argument(
    '-e',
    '--evaluate',
    type=str,
    metavar='FILE',
    help='evaluate model FILE on validation set')

parser.add_argument(
    '--seed', 
    default=0, 
    type=int, 
    help='random seed, set to 0 to disable')

parser.add_argument(
    '--model',
    '-a',
    metavar='MODEL',
    default='resnet20',
    help='model architecture ')

parser.add_argument(
    '--dataset',
    default='cifar10',
    type=str,
    help='dataset, default: cifar10')

parser.add_argument(
    '--data_path',
    type=str,
    default='/home/data',
    help='The dictionary where the dataset is stored.')

parser.add_argument(
    '--type',
    default='torch.cuda.FloatTensor',
    help='type of tensor - e.g torch.cuda.FloatTensor')

parser.add_argument(
    '--gpus',
    default='0',
    help='gpus used for training - e.g 0,1,2,3')

parser.add_argument(
    '--lr', 
    default=0.1, 
    type=float, 
    help='learning rate')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='Weight decay of loss. default: 5e-4')

parser.add_argument(
    '--momentum',
    default=0.9, 
    type=float, 
    help='momentum')

parser.add_argument(
    '--workers',
    default=8,
    type=int,
    help='number of data loading workers (default: 8)')

parser.add_argument(
    '--epochs',
    default=600,
    type=int,
    help='number of total epochs to run')

parser.add_argument(
    '--start_epoch',
    default=-1,
    type=int,
    help='manual epoch number (useful on restarts)')

parser.add_argument(
    '-b',
    '--batch_size',
    default=256,
    type=int,
    help='mini-batch size for training (default: 256)')

parser.add_argument(
    '-bt',
    '--batch_size_test',
    default=128,
    type=int,
    help='mini-batch size for testing (default: 128)')

parser.add_argument(
    '--print_freq',
    '-p',
    default=100,
    type=int,
    help='print frequency (default: 100)')

parser.add_argument(
    '--time_estimate',
    default=1,
    type=int,
    help='print estimating finish time, set to 0 to disable')

parser.add_argument(
    '--lr_type',
    type=str,
    default='cos',
    help='choose lr_scheduler, (default:cos)')

parser.add_argument(
    '--lr_decay_step',
    nargs='+',
    type=int,
    help='lr decay step for MultiStepLR')

parser.add_argument(
    '--optimizer',
    type=str,
    default='sgd',
    help='choose optimizer, (default:sgd)')

parser.add_argument(
    '--warm_up',
    dest='warm_up',
    action='store_true',
    help='use warm up or not')

parser.add_argument(
    '--tau_min',
    default=0.85, 
    type=float, 
    help='tau_min')

parser.add_argument(
    '--tau_max',
    default=0.99, 
    type=float, 
    help='tau_max')

parser.add_argument(
    '--wbit',
    default=1,
    type=int,
    help='weight precision')

parser.add_argument(
    '--abit',
    default=1,
    type=int,
    help='activation precision')

parser.add_argument(
    '--nosaves',
    dest='nosaves',
    action='store_true',
    help='No saves')

parser.add_argument(
    '--refit',
    type=str,
    metavar='BALANCED',
    help='refit the balanced models for new unbalanced surrogate gradient function')

parser.add_argument(
    '--delta',
    default=1, 
    type=float, 
    help='delta')

parser.add_argument(
    '--theta',
    default=0, 
    type=float, 
    help='theta')

parser.add_argument(
    '--rho',
    default=0.3, 
    type=float, 
    help='rho')

parser.add_argument(
    '--phi',
    default=1, 
    type=float, 
    help='phi')

parser.add_argument(
    '--min_val',
    default=-1, 
    type=float, 
    help='min_val')

parser.add_argument(
    '--max_val',
    default=1, 
    type=float, 
    help='max_val')

parser.add_argument(
    '--sg',
    type=str,
    default='s_lste',
    help='surrogate gradient function (default:ste)')


args = parser.parse_args()