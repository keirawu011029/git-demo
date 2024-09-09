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