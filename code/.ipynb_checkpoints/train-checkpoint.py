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

# '''
# pip install numpy numba Cython asteroid sagemaker-inference PyYAML pytorch-lightning==1.4.9
# '''



from os import listdir
from os.path import isfile, join
import sys
import textwrap
# from sagemaker_inference import content_types, decoder, default_inference_handler, encoder

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
# from asteroid.data import LibriMix
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.engine.optimizers import make_optimizer
from asteroid.models import DPRNNTasNet
from asteroid.utils import prepare_parser_from_dict
from asteroid.utils import parse_args_as_dict
from wham_dataset_no_sf import *
import json
import argparse




def _train():#args
    
    with open('./code/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    pp = argparse.ArgumentParser(conflict_handler='resolve')
    parser = prepare_parser_from_dict(def_conf, parser=pp)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    conf = arg_dic
    
    train_dir = conf['data']['train_dir']
    val_dir = conf['data']['valid_dir']
    
    train_set = WhamDataset_no_sf(train_dir, conf['data']['task'],
                            sample_rate=conf['data']['sample_rate'], segment=conf['data']['segment'],
                            nondefault_nsrc=conf['data']['nondefault_nsrc'])
    val_set = WhamDataset_no_sf(val_dir, conf['data']['task'], segment=conf['data']['segment'],
                          sample_rate=conf['data']['sample_rate'], nondefault_nsrc=conf['data']['nondefault_nsrc'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)

    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)
    

    conf['masknet'].update({'n_src': train_set.n_src})

    model = DPRNNTasNet(**conf['filterbank'], **conf['masknet'])
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    # Define scheduler
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                      patience=5)

    
    exp_dir = conf['data']['model_dir']#args.model_dir
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    system = System(model=model, loss_func=loss_func, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    scheduler=scheduler, config=conf)
    system.batch_size = 1

    gpus = -1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(max_epochs=conf['training']['epochs'],
                         default_root_dir=exp_dir,
                         gpus=gpus,
                         distributed_backend='ddp',
                         gradient_clip_val=conf['training']["gradient_clipping"])
    trainer.fit(system)
#     system.cpu()

    to_save = system.model.serialize()
    torch.save(to_save, os.path.join(exp_dir, 'best_model.pth'))



if __name__ == '__main__':
    


    _train() #vars(parser.parse_args())

