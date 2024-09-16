#!/usr/bin/env python

import os
import numpy as np
from collections import OrderedDict

import torch

import src.config as config
from src.data import get_csv_sample_rate, get_dataset_from_configs, collate_into_list, get_loss_weights_and_flags, get_sample_rate
from src.model.model_utils import get_model
from src.train import Trainer

def load_12ECG_model(output_training_directory):
    # load the model from disk
    data_cfg = config.DataConfig("config/data.json")
    preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
    model_cfg = config.ModelConfig("config/model.json")
    run_cfg = config.RunConfig("config/run.json")

    models, thresholds = [], []
    for fold in range(10):
        model, _ = get_model(model_cfg, 1, len(data_cfg.scored_classes))  # Change num_channels to 1
        checkpoint = torch.load(os.path.join(output_training_directory, 'finalized_model_%d.sav' % fold),
                                map_location=torch.device("cpu"))
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
        models.append(model)

        threshold = np.load(os.path.join(output_training_directory, 'finalized_model_thresholds_%d.npy' % fold))
        thresholds.append(threshold)

    thresholds = np.average(np.stack(thresholds, axis=0), axis=0)
    eval_list = [data_cfg, preprocess_cfg, run_cfg, models, thresholds]

    return eval_list

def filter_signal(x, preprocess_cfg, sample_rate):
    """ filter ecg signal """
    nyq = sample_rate * 0.5
    for i in range(len(x)):
        for cutoff in preprocess_cfg.filter_highpass:
            x[i] = sig.filtfilt(*sig.butter(2, cutoff / nyq, btype='highpass'), x[i])
        for cutoff in preprocess_cfg.filter_lowpass:
            if cutoff >= nyq: cutoff = nyq - 0.05
            x[i] = sig.filtfilt(*sig.butter(2, cutoff / nyq, btype='lowpass'), x[i])
        for cutoff in preprocess_cfg.filter_bandpass:
            x[i] = sig.filtfilt(*sig.butter(2, [cutoff[0] / nyq, cutoff[1] / nyq], btype='bandpass'), x[i])
        for cutoff in preprocess_cfg.filter_notch:
            x[i] = sig.filtfilt(*sig.iirnotch(cutoff, cutoff, sample_rate), x[i])

    return x


def scale_signal(x, preprocess_cfg):
    """ scale ecg signal """
    for i in range(len(x)):
        if preprocess_cfg.scaler is None: continue
        elif "minmax" in preprocess_cfg.scaler:   scaler = MinMaxScaler()
        elif "standard" in preprocess_cfg.scaler: scaler = StandardScaler()
        elif "robust" in preprocess_cfg.scaler:   scaler = RobustScaler()
        scaler.fit(np.expand_dims(x[i], 1))
        x[i] = scaler.transform(np.expand_dims(x[i], 1)).squeeze()

    return x

def preprocess_signal(x, preprocess_cfg, sample_rate):
    """ resample, filter, scale, ecg signal """
    target_length = preprocess_cfg.target_length
    target_sample_rate = preprocess_cfg.sample_rate
    
    # Resample to target sample rate
    if sample_rate != target_sample_rate:
        num = int(x.shape[1] * (target_sample_rate / sample_rate))
        x = sig.resample(x, num, axis=1)
    
    # Pad or truncate to target length
    current_length = x.shape[1]
    if current_length < target_length:
        pad_width = ((0, 0), (0, target_length - current_length))
        x = np.pad(x, pad_width, mode='constant', constant_values=0)
    elif current_length > target_length:
        x = x[:, :target_length]
    
    x = filter_signal(x, preprocess_cfg, target_sample_rate)
    x = scale_signal(x, preprocess_cfg)
    return x
def run_12ECG_classifier(data, header, eval_list):
    data_cfg, preprocess_cfg, run_cfg, models, thresholds = eval_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, np.ndarray):  # MAT file data
        data_cfg.data = data[1:2]  # Select only the 2nd lead
        sample_rate = get_sample_rate(header)
    else:  # CSV file data
        data_cfg.data = data.reshape(1, -1)  # Reshape to (1, sequence_length)
        sample_rate = get_csv_sample_rate(data_cfg.data.shape[1])
    
    data_cfg.header = header
    x = preprocess_signal(data_cfg.data, preprocess_cfg, sample_rate)
    dataset_val = ecg_dataset_for_inference([torch.from_numpy(x).float()], [torch.zeros(len(data_cfg.scored_classes))])
    iterator_val = torch.utils.data.DataLoader(dataset_val, 1, collate_fn=collate_into_list)


    outputs_list = []
    for model in models:
        loss_weights_and_flags = get_loss_weights_and_flags(data_cfg, run_cfg)
        trainer = Trainer(model, data_cfg, run_cfg.multilabel, loss_weights_and_flags)
        trainer.set_device(device, data_parallel=False)

        for B, batch in enumerate(iterator_val):
            trainer.evaluate(batch)

        outputs = trainer.logger_eval.scalar_outputs[0][0]
        outputs_list.append(outputs)

    classes = data_cfg.scored_classes
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    for i in range(num_classes):
        for j in range(len(outputs_list)):
            current_score[i] += outputs_list[j][i]
        current_score[i] = current_score[i] / len(outputs_list)
        if current_score[i] > thresholds[i]: current_label[i] = 1

    return current_label, current_score, classes