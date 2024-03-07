import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import math
import time
import logging
import json
import copy
import random
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
# from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BertTokenizer

from train_reqa import prepare_model, test


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_dataloaders(args):
    from utils_data import ReQADataset as Dataset
    # initialize datasets
    test_question_dataset = Dataset(args, split="test", data_type="question")
    test_candidate_dataset = Dataset(args, split="test", data_type="candidate")

    logger.info(f"test question size: {test_question_dataset.__len__()}")
    logger.info(f"test candidate size: {test_candidate_dataset.__len__()}")

    test_question_data_loader = torch.utils.data.DataLoader(
        test_question_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_question_dataset.collate_fn)

    test_candidate_data_loader = torch.utils.data.DataLoader(
        test_candidate_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_candidate_dataset.collate_fn)

    return test_question_data_loader, test_candidate_data_loader

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument('--batch_size', type=int)
    # evaluation
    parser.add_argument("--main_metric", type=str)
    # model
    parser.add_argument('--model_name_or_paths', type=str, nargs='+')
    parser.add_argument('--pooler', type=str)
    parser.add_argument("--matching_func", type=str)
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--save_results', type=str)

    args = parser.parse_args()


    # set path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger.info(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = dict()

    for model_name_or_path in args.model_name_or_paths:
        args.model_name_or_path = model_name_or_path

        test_question_data_loader, test_candidate_data_loader = prepare_dataloaders(args)
        model = prepare_model(args)

        results[model_name_or_path] = test(model, test_question_data_loader, test_candidate_data_loader, args)



    if args.save_results == "True":
        with open(args.save_dir + "/results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()