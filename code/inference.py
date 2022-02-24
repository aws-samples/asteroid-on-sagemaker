# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import


import argparse
import os
os.system('pip install numpy==1.18.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/')
os.system('pip install numba==0.48 -i https://pypi.tuna.tsinghua.edu.cn/simple/')
os.system('pip install pytorch-lightning==0.7.6 -i https://pypi.tuna.tsinghua.edu.cn/simple/')
os.system('pip install Cython==0.29.15 -i https://pypi.tuna.tsinghua.edu.cn/simple/')
os.system('pip install asteroid==0.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/')
os.system('pip install sagemaker-inference==1.3.2.post0 -i https://pypi.tuna.tsinghua.edu.cn/simple/')
os.system('pip install PyYAML==5.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/')

# numpy==1.18.1
# numba==0.48
# pytorch-lightning==0.7.6
# Cython==0.29.15
# asteroid==0.3.0
# sagemaker-inference==1.3.2.post0
# PyYAML==5.3.1

from os import listdir
from os.path import isfile, join
import sys
import textwrap
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder

import numpy as np
import torch
from six import BytesIO
import yaml
import sys
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.engine.optimizers import make_optimizer
from conv_tasnet import DPRNNTasNet
from asteroid.utils import prepare_parser_from_dict
from asteroid.utils import parse_args_as_dict
from wham_dataset_no_sf import *
import json
import argparse




def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'best_model.pth')
    model = DPRNNTasNet.from_pretrained(model_path)

    return model


def predict_fn(input_data, model):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()
    model.cuda()
    with torch.no_grad():
        estimate_source = model(input_data)  # [B, C, T]

    return estimate_source
