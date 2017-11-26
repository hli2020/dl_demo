import argparse
from train import *
from dataset import *

parser = argparse.ArgumentParser(description='DNI')
parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--model_type', choices=['mlp', 'cnn'], default='mlp', help='currently support mlp and cnn')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--conditioned', type=bool, default=True)
parser.add_argument('--plot', type=bool, default=True)
parser.add_argument('--use_gpu', type=bool, default=False)
args = parser.parse_args()

# do not support using mlp to trian cifar
assert args.dataset != 'cifar10' or args.model_type != 'mlp'
model_name = '%s.%s_dni' % (args.dataset, args.model_type, )
if args.conditioned:
    model_name += '.conditioned'
args.model_name = model_name
if args.dataset == 'mnist':
    data = mnist(args)
elif args.dataset == 'cifar10':
    data = cifar10(args)
m = classifier(args, data)  # init the model; in the "train.py"
m.train_model()             # the overall structure of the code system; in the "train.py"
