#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import time
import os
import shutil
import json
import jsonlines
import torch
import numpy as np

from sklearn.decomposition import PCA
from tqdm import tqdm
from selection_methods.utils_model_selection import whitening, random_candidate_sampling, BM25_candidate_sampling

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def obtain_train_embeddings(model, train_data_loader):
    model.eval()
    with torch.no_grad():
        src_embeddings = []
        tgt_embeddings = []
        src_ids = []
        for batch in tqdm(train_data_loader, desc='[encoding training src-tgt pairs]', leave=False):
            # forward
            batch_src_embeddings, batch_tgt_embeddings, _ = model(**batch)
            src_embeddings.append(batch_src_embeddings.cpu().numpy())
            tgt_embeddings.append(batch_tgt_embeddings.cpu().numpy())
            src_ids.append(batch["src_ids"].cpu().numpy())
        src_embeddings = np.concatenate(src_embeddings, 0)
        tgt_embeddings = np.concatenate(tgt_embeddings, 0)
        src_ids = np.concatenate(src_ids, 0)

    return src_embeddings, tgt_embeddings, src_ids


def dataset_encoding(args):
    cached_dir = f"./cached_data/{args.dataset}"
    if args.aggregate:
        cached_dir = cached_dir + '_aggregate'
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    plm_name = [s for s in args.model_name_or_path.split('/') if s !=''][-1]
    cached_dataset_embedding_file = os.path.join(cached_dir, f'{plm_name}_embedding')
    # load processed dataset or process the original dataset
    if os.path.exists(cached_dataset_embedding_file):
        logging.info("Loading encoded dataset from cached file %s", cached_dataset_embedding_file)
        data_dict = torch.load(cached_dataset_embedding_file)
        train_src_embeddings = data_dict["train_src_embeddings"]
        train_tgt_embeddings = data_dict["train_tgt_embeddings"]
        train_src_ids = data_dict["train_src_ids"]
    else:
        # load dataset
        from utils_data import ReQADataset as Dataset
        # initialize datasets
        train_dataset = Dataset(args, split="train")
        
        logging.info(f"train data size: {train_dataset.__len__()}")

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=train_dataset.collate_fn)


        # preparing model
        from models.dual_encoder import RankModel
        model = RankModel(args)
        model.to(args.device)
        train_src_embeddings, train_tgt_embeddings, train_src_ids = obtain_train_embeddings(model, train_data_loader)
        # test_question_embeddings, test_candidate_embeddings, test_ground_truths = obtain_test_embeddings(model, test_question_data_loader, test_candidate_data_loader)

        saved_data = {
            "train_src_embeddings": train_src_embeddings,
            "train_tgt_embeddings": train_tgt_embeddings,
            "train_src_ids": train_src_ids,
        }
                
        logging.info("Saving encoded dataset to %s", cached_dataset_embedding_file)
        torch.save(saved_data, cached_dataset_embedding_file)
    return train_src_embeddings, train_tgt_embeddings, train_src_ids




def main(args: argparse.Namespace):
    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name_or_path in args.model_name_or_paths:
        args.model_name_or_path = model_name_or_path
        embeddings_dict = {}

        train_src_embeddings, train_tgt_embeddings, train_src_ids = dataset_encoding(args)
        # continue
        whitened_train_src_embeddings, whitened_train_tgt_embeddings = whitening(np.copy(train_src_embeddings), np.copy(train_tgt_embeddings))

        for candidate_size in args.all_candidate_sizes:
            args.candidate_size = str(candidate_size)
            
            for method in args.methods:
                args.method = method
                TransMetric = None
                logging.info(f"{args.method} (candidate size {args.candidate_size}) for embeddings from {args.model_name_or_path} on dataset {args.dataset}.")

                if args.method.startswith("ZeroShotProxy"):
                    from selection_methods.zero_shot_proxy import ZeroShotProxy as TransMetric
                elif args.method.startswith("Rreg"):
                    from selection_methods.rank_linear import Rank_Reg as TransMetric
                elif args.method.startswith("TMI"):
                    from selection_methods.tmi import TMI as TransMetric
                elif args.method.startswith("NLEEP"):
                    from selection_methods.nleep import NLEEP as TransMetric
                elif args.method.startswith("Logistic"):
                    from selection_methods.logistic import Logistic as TransMetric
                elif args.method.startswith("HScoreR"):
                    from selection_methods.hscore_reg import HScoreR as TransMetric
                elif args.method.startswith("HScore"):
                    from selection_methods.hscore import HScore as TransMetric
                elif args.method.startswith("LogME"):
                    from selection_methods.logme import LogME as TransMetric
                elif args.method.startswith("MixupLogME"):
                    from selection_methods.mixuplogme import LogME as TransMetric
                elif args.method.startswith("RLogME"):
                    from selection_methods.rank_logme import LogME as TransMetric
                elif args.method.startswith("GBC"):
                    from selection_methods.gbc import GBC as TransMetric
                elif args.method.startswith("TransRate"):
                    from selection_methods.transrate import TransRate as TransMetric
                elif args.method.startswith("SFDA"):
                    from selection_methods.sfda import SFDA as TransMetric
                elif args.method.startswith("PACTran"):
                    from selection_methods.pactran import PACTran as TransMetric

                if eval(args.save_results):
                    save_dir = f"output/{args.dataset}/model_selection"
                    if args.aggregate:
                        save_dir = save_dir + '_aggregate'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    results_file = f"{save_dir}/{args.method}.json"
                    if os.path.exists(results_file):
                        with open(results_file, "r") as f:
                            results_dict = json.load(f)
                    else:
                        results_dict = dict()


                if eval(args.save_results) and args.candidate_size in results_dict\
                    and args.model_name_or_path in results_dict[args.candidate_size] and not eval(args.overwrite_results):
                    logging.info("Skipping the candidate model already has been scored.")
                    logging.info("-------------------------------END-------------------------------\n")
                    continue
                else:
                    all_scores = []
                    all_times = []
                    for seed in args.seeds:
                        args.seed = int(seed)
                        logging.info(f"running by seed {args.seed}...")
                        
                        metric = TransMetric(args)

                        embedding_key = f"whitening_random_{args.seed}_{args.candidate_size}" if "whitening" in args.method else f"random_{args.seed}_{args.candidate_size}"

                        if embedding_key in embeddings_dict:
                            features, labels = embeddings_dict[embedding_key]
                        else:
                            if args.candidate_sampling_method == 'BM25':
                                repeated_src_embeddings, sampled_candidate_embeddings = BM25_candidate_sampling(np.copy(whitened_train_src_embeddings) if "whitening" in args.method else np.copy(train_src_embeddings),
                                                                                                                np.copy(whitened_train_tgt_embeddings) if "whitening" in args.method else np.copy(train_tgt_embeddings),
                                                                                                                np.copy(train_src_ids),
                                                                                                                args)
                            else:
                                repeated_src_embeddings, sampled_candidate_embeddings = random_candidate_sampling(np.copy(whitened_train_src_embeddings) if "whitening" in args.method else np.copy(train_src_embeddings),
                                                                                                                  np.copy(whitened_train_tgt_embeddings) if "whitening" in args.method else np.copy(train_tgt_embeddings),
                                                                                                                  np.copy(train_src_ids),
                                                                                                                  args)
                            
                            features = (repeated_src_embeddings * sampled_candidate_embeddings).reshape(-1, repeated_src_embeddings.shape[-1])

                            labels = []
                            for i in range(repeated_src_embeddings.shape[0]):
                                labels.extend([1] + [0]*(int(args.candidate_size)-1))
                            labels = np.array(labels)

                            embeddings_dict[embedding_key] = [features, labels]

                        if args.aggregate:
                            semantic_features = features[:,:-args.agg_dim]
                            semantic_score, semantic_score_time = metric.score(np.copy(semantic_features), np.copy(labels))

                            lexical_features = features[:,-args.agg_dim:]
                            lexical_score, lexical_score_time = metric.score(np.copy(lexical_features), np.copy(labels))

                            # print(f'{semantic_score} + a * {lexical_score}')

                            score = semantic_score + args.agg_alpha * lexical_score
                            score_time = semantic_score_time + lexical_score_time
                        else:
                            score, score_time = metric.score(np.copy(features), np.copy(labels))
                    
                        all_scores.append(score)
                        all_times.append(score_time)

                if eval(args.save_results):
                    results_dict[args.candidate_size] = results_dict.get(args.candidate_size, dict())
                    results_dict[args.candidate_size][args.model_name_or_path] = {
                        "all_scores": all_scores,
                        "all_times": all_times,
                    }
                    with open(results_file, "w") as f:
                        json.dump(results_dict, f, indent=4)

                
                logging.info(f"all scores: {all_scores}")
                logging.info(f"all times: {all_times}")
                # logging.info(f"avg score: {np.mean(all_scores)}, avg time: {round(np.mean(all_times), 4)}s")
                logging.info("-------------------------------END-------------------------------\n")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Framework for Model Selection')

    parser.add_argument('--methods', nargs='+', help='List of Model selection method.')
    parser.add_argument('--dataset', type=str, nargs='?', help='Dataset from the HuggingFace Dataset library.')
    parser.add_argument('--save_results', type=str, nargs='?', help='Whether to save results.')
    parser.add_argument('--model_name_or_paths', nargs='+', help='list of pretrained language model identifiers.')
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--overwrite_results', type=str, default="False")
    parser.add_argument("--matching_func", type=str)
    parser.add_argument('--pooler', type=str, nargs='?', help='pooling strategy for sentence classification (default: None)')
    parser.add_argument('--all_candidate_sizes', nargs='+', help='list of candidate sizes.')
    parser.add_argument('--candidate_sampling_method', type=str, default="random")
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--agg_dim', type=int, default=768, help='dimension of lexical embeddings for aggregation (default: 768)')
    parser.add_argument('--agg_alpha', type=float, default=1.0, help='coefficient for aggregation score calculation (default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
    parser.add_argument('--seeds', nargs='+', help='list of random seeds')

    main(parser.parse_args())
