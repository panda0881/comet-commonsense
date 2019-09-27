import os
import sys
import argparse
import torch

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
from tqdm import tqdm
import ujson as json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="/data/hzhangal/comet/pretrained_models/conceptnet_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    data_for_predict = list()
    with open('test.txt', 'r') as f:
        for line in f:
            tmp_words = line[:-1].split('\t')
            data_for_predict.append((tmp_words[0], tmp_words[1]))

    prediction_result = list()
    sampler = interactive.set_sampler(opt, 'beam-10', data_loader)
    for tmp_pair in tqdm(data_for_predict):
        tmp_event = tmp_pair[1]
        tmp_relation = tmp_pair[0]
        tmp_sampler = 'beam-10'
        outputs = interactive.get_conceptnet_sequence(
            tmp_event, model, sampler, data_loader, text_encoder, tmp_relation, print_result=False)
        prediction_result.append(outputs)

    with open('original_result.json', 'w') as f:
        json.dump(prediction_result, f)


    print('end')

