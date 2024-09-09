import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from data import get_CIFAR10_loader
from attacks.AdversarialInput.SharpnessAware2 import *
from models import *

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models 
from tester import test_multimodel_acc_one_image, test_transfer_attack_acc, \
    test_transfer_attack_acc_and_cosine_similarity, test_transfer_attack_acc_swag, test_transfer_attack_acc_final, test_transfer_attack_acc_cifar10
import timm

import matplotlib.pyplot as plt
import argparse
import timm
from timm.models import create_model
import models.resnet_at_models.advresnet_gbn_gelu as advres
from models.resnet_at_models.EightBN import EightBN
from models.resnet_at_models.advresnet_gbn_gelu import ResNet_AT
from timm.models.xcit import Xcit
import json
import models.cifar_models as cifar_models


# from defenses import Randomization, JPEGCompression, BitDepthReduction, \
#     NeuralRepresentationPurifier, randomized_smoothing_resnet50


parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str, default='MI_RAP_every_momentum')
parser.add_argument("--model_dir", type=str, default='./models')
parser.add_argument("--result_dir", type=str, default='./result_cifar10/find_best_vit_cifar10')
parser.add_argument("--result_file_name", type=str, default=None)
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--targerted_attack', action='store_true')
parser.add_argument('--model_num', type=int, default=40)
parser.add_argument('--total_step', type=int, default=1)
parser.add_argument('--step_size', type=float, default=2/255)
parser.add_argument('--reverse_step_size', type=float, default=0.1/255)
parser.add_argument('--reverse_step', type=int, default=5)
parser.add_argument('--late_start', type=int, default=15)
parser.add_argument('--inner_step_size', type=int, default=250)
parser.add_argument("--model_archs", nargs='+',default='resnetat xcitat')

parser.add_argument('--vit_model_dir', type=str, default='')



args = parser.parse_args()

# FIXME: 
loader = get_CIFAR10_loader(batch_size=1000, targeted=args.targerted_attack)

model_cfgs = []
white_models = []

# resnet_at
backbone = wideresnetwithswish('wrn-70-16-swish', dataset='cifar10s', num_classes=10, device=args.device)
model = torch.nn.Sequential(backbone)
if 'resnetat' in args.model_archs:
    model_cfgs.append(model)
white_models.append(model)

# vit
model = timm.create_model('vit_base_patch16_224', num_classes=10)

# resnet50
model = cifar_models.resnet50()
if 'resnet' in args.model_archs:
    model_cfgs.append(model)
white_models.append(model)


attacker_list = [eval(args.attack)]
for now_attacker in attacker_list:
    attacker = now_attacker(model=model_cfgs,
                            total_step=args.total_step,
                            step_size=args.step_size,
                            reverse_step_size=args.reverse_step_size,
                            reverse_step=args.reverse_step,
                            late_start=args.late_start,
                            inner_step_size=args.inner_step_size,
                            targeted_attack=args.targerted_attack)
    print(attacker.__class__)
    print(attacker.total_step)
    test_transfer_attack_acc_cifar10(args, attacker, loader, white_models)


