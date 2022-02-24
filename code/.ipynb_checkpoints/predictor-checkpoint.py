import os
import sys
import joblib
import glob
import json
import time
import numpy
import flask
from flask import Flask, Response
import logging
import numpy as np
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.engine.optimizers import make_optimizer
from asteroid.models import DPRNNTasNet
from asteroid.utils import prepare_parser_from_dict
from asteroid.utils import parse_args_as_dict

import torch

# set to true to print incoming request headers and data
DEBUG_FLAG = False

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

app.logger.info('CPU Model Serving Workflow')
app.logger.info(f'> {os.cpu_count()} CPUs detected \n')

@app.route("/ping", methods=["GET"])
def ping():
    """ SageMaker required method, ping heartbeat """
    return Response(response="\n", status=200)

@app.route("/invocations", methods=["POST"])
def predict_fn():
    """
    Run CPU or GPU inference on input data,
    called everytime an incoming request arrives
    """
    # parse user input
    try:
        if DEBUG_FLAG:
            app.logger.debug(flask.request.headers)
            app.logger.debug(flask.request.content_type)
            app.logger.debug(flask.request.get_data())
        raw = flask.request.data.decode('utf-8')
        raw = json.loads(raw.replace('\'','"'))
        print('raw',type(raw))
        raw=raw['data']
        input_data = torch.Tensor(np.array(raw))
        print('input_data',type(input_data))
#         os.system('ls /opt/ml/model/')
        print(1)
        model_path = glob.glob('/opt/ml/model/best_model.pth')[0]
        model = DPRNNTasNet.from_pretrained(model_path)
        model.eval()
        print(2)
        with torch.no_grad():
            estimate_source = model(input_data)  # [B, C, T]
        dictionary = {'response':estimate_source.numpy().tolist()}
        print(3,estimate_source.numpy().tolist())
        _payload = json.dumps(dictionary)
        return flask.Response(response=_payload, status=200, mimetype='application/json')

    except Exception as e:
        return Response(
            response="error",
            status=415,
            mimetype='text/csv'
        )
