'''
This script handling the training process.
'''
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



logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    logger.info(f"Current random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_epoch(model, train_data_loader, optimizer, scheduler, scaler, epoch_i, args):

    model.train()
    total_tr_loss = 0.0
    total_train_batch = 0
    total_acc = 0.0
    start = time.time()

    for step, batch in enumerate(tqdm(train_data_loader, desc='  -(Training)', leave=False)):

        # forward
        if args.mixed_training:
            with autocast():
                tr_loss, tr_acc = model(**batch)
        else:
            tr_loss, tr_acc = model(**batch)
        
        # backward
        if args.mixed_training:
            scaler.scale(tr_loss).backward()
        else:
            tr_loss.backward()

        # record
        total_acc += tr_acc
        total_tr_loss += tr_loss.item()
        total_train_batch += 1

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if args.mixed_training:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        model.zero_grad()
    
    logger.info('[Epoch{epoch: d}] - (Train) Loss ={train_loss: 8.5f}, Acc ={acc: 3.2f} %, '\
                'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i, 
                                                    train_loss=total_tr_loss / total_train_batch, 
                                                    acc=100 * total_acc / total_train_batch,
                                                    elapse=(time.time()-start)/60))

def dev_epoch(model, dev_data_loader, epoch_i, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    start = time.time()

    total_dev_acc = []
    question_embeddings = []
    answer_embeddings = []
    all_src_ids = []

    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc='  -(Dev)', leave=False):
            # forward
            if args.dev_metric == "p1":
                question_embedding, answer_embedding, src_ids = model(**batch)
                question_embeddings += [question_embedding.cpu()]
                answer_embeddings += [answer_embedding.cpu()]
                all_src_ids += [src_ids.cpu()]
            elif args.dev_metric == "acc":
                _, acc = model(**batch)
                total_dev_acc += [acc]

    if args.dev_metric == "p1":
        question_embeddings = torch.cat(question_embeddings, 0)
        answer_embeddings = torch.cat(answer_embeddings, 0)
        all_src_ids = torch.cat(all_src_ids, 0)

        predict_logits = model.matching(question_embeddings, answer_embeddings, all_src_ids, False)
        predict_result = torch.argmax(predict_logits, dim=1)
        labels = torch.arange(0, predict_logits.shape[0])
        dev_p1 = labels == predict_result
        dev_p1 = (dev_p1.int().sum() / (predict_logits.shape[0] * 1.0)).item() * 100

        logger.info('[Epoch{epoch: d}] - (Dev  ) P@1 ={p1: 3.2f} %, '\
                    'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i, 
                                                        p1=dev_p1,
                                                        elapse=(time.time()-start)/60))
        return dev_p1
    elif args.dev_metric == "acc":
        dev_acc = np.mean(total_dev_acc)*100
        logger.info('[Epoch{epoch: d}] - (Dev  ) Acc ={acc: 3.2f} %, '\
                    'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i, 
                                                        acc=dev_acc,
                                                        elapse=(time.time()-start)/60))
        return dev_acc                            
    


class NumpyWhitening():
    def __init__(self):
        pass
    
    def fit(self, sentence_embeddings):
        self.mu = sentence_embeddings.mean(axis=0, keepdims=True).astype(np.float32)
        cov = np.cov(sentence_embeddings.T)
        u, s, vh = np.linalg.svd(cov)
        self.W = np.dot(u, np.diag(1 / np.sqrt(s))).astype(np.float32)
        self.inverse_W = np.dot(np.diag(np.sqrt(s)), u.T).astype(np.float32)
    
    def transform(self, vecs):
        return (vecs - self.mu).dot(self.W)

    def inverse_transform(self, vecs):
        return vecs.dot(self.inverse_W) + self.mu


class TorchWhitening():
    def __init__(self):
        pass
    
    def fit(self, sentence_embeddings):
        self.mu = torch.mean(sentence_embeddings, dim=0, keepdims=True)
        cov = (sentence_embeddings-self.mu).t().mm((sentence_embeddings-self.mu)) / (sentence_embeddings.shape[0]-1)
        u, s, v = torch.svd(cov)
        self.W = u.mm(torch.diag(1 / torch.sqrt(s)))
    
    def transform(self, vecs):
        return (vecs - self.mu).mm(self.W)



def obtain_whitening_params(model, data_loaders, args):
    model.eval()
    sentence_embeddings = []

    with torch.no_grad():
        for data_loader in data_loaders:
            # rebuilt the dataloader without shuffling which removes randomness
            data_loader = torch.utils.data.DataLoader(
                data_loader.dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=data_loader.dataset.collate_fn)
            for batch in tqdm(data_loader, desc='[sentence encoding for whitening]', leave=False):
                # forward
                src_embedding, tgt_embedding, _ = model(**batch)
                sentence_embeddings.append(src_embedding.cpu())
                sentence_embeddings.append(tgt_embedding.cpu())
    
    sentence_embeddings = torch.cat(sentence_embeddings, 0)

    whitening = TorchWhitening()
    whitening.fit(sentence_embeddings)

    # from sklearn.decomposition import PCA
    # pca_whitening = PCA(n_components=sentence_embeddings.shape[1], whiten=True).fit(sentence_embeddings)

    # import pdb; pdb.set_trace()
    
    return whitening


def whiten_sentence_embeddings(question_embeddings, candidate_embeddings, whitening, args):
    return whitening.transform(question_embeddings), whitening.transform(candidate_embeddings)


def obtain_test_embeddings(model, test_question_data_loader, test_candidate_data_loader, args):
    model.eval()
    with torch.no_grad():
        question_embeddings = []
        test_ground_truth = []
        for batch in tqdm(test_question_data_loader, desc='[encoding test questions]', leave=False):
            # forward
            test_ground_truth += list(batch["ground_truth"])
            del batch["ground_truth"]
            question_embedding = model.sentence_encoding(**batch)
            question_embeddings.append(question_embedding.cpu())
        question_embeddings = torch.cat(question_embeddings, 0)

        candidate_embeddings = []
        for batch in tqdm(test_candidate_data_loader, desc='[encoding test candidates]', leave=False):
            # forward
            candidate_embedding = model.sentence_encoding(**batch)
            candidate_embeddings.append(candidate_embedding.cpu())
        candidate_embeddings = torch.cat(candidate_embeddings, 0)

    return question_embeddings, candidate_embeddings, test_ground_truth


def test(model, test_question_data_loader, test_candidate_data_loader, args):

    question_embeddings, candidate_embeddings, test_ground_truth = obtain_test_embeddings(model, test_question_data_loader, test_candidate_data_loader, args)

    predict_logits = model.matching(question_embeddings, candidate_embeddings, None, False).numpy()

    p_counts = {1: 0.0}
    r_counts = {5: 0.0, 10: 0.0}
    mrr_counts = 0

    for idx in range(len(test_ground_truth)):
        pred = np.argsort(-predict_logits[idx]).tolist()

        # precision at K
        for rank in p_counts.keys():
            numerator = 0.0
            denominator = rank
            for gt in test_ground_truth[idx]:
                if numerator == rank:
                    break
                if gt in pred[:rank]:
                    numerator += 1
            p_counts[rank] += numerator / denominator
        
        # recall at K
        denominator = len(test_ground_truth[idx])
        for rank in r_counts.keys():
            numerator = 0.0
            for gt in test_ground_truth[idx]:
                if numerator == rank:
                    break
                if gt in pred[:rank]:
                    numerator += 1
            r_counts[rank] += numerator / denominator
                    
        # mrr
        for r, p in enumerate(pred):
            if p in test_ground_truth[idx]:
                mrr_counts += 1 / (r + 1)
                break

    mrr = np.round(mrr_counts / len(test_ground_truth), 4) * 100
    p_at_k = {k: np.round(v / len(test_ground_truth), 4) * 100 for k, v in p_counts.items()}
    r_at_k = {k: np.round(v / len(test_ground_truth), 4) * 100 for k, v in r_counts.items()}

    # logger.info('[Seed: {seed:d}] - (Test ) MRR ={mrr: 3.2f} %, P@1 ={p1: 3.2f} %,'\
    #               ' R@5 ={r5: 3.2f} %, R@10 ={r10: 3.2f} %'.format(seed=args.seed,
    #                                                                mrr=mrr,
    #                                                                p1=p_at_k[1],
    #                                                                r5=r_at_k[5],
    #                                                                r10=r_at_k[10]))

    return {
        "mrr": mrr,
        "p1": p_at_k[1],
        "r5": r_at_k[5],
        "r10": r_at_k[10]
    }


def run(model, train_data_loader, test_question_data_loader, test_candidate_data_loader, args, results):
    # Prepare optimizer and schedule (linear warmup and decay)
    args.num_train_instances = train_data_loader.dataset.__len__()
    args.num_training_steps =  math.ceil(args.num_train_instances / args.batch_size) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=math.ceil(args.warmup_proportion * args.num_training_steps), 
        num_training_steps=args.num_training_steps
    )

    # mixed precision training
    scaler = GradScaler()
    
    # record best performance
    best_metric = 0
    best_epoch = 0

    # start training
    for epoch_i in range(1, args.epoch+1):
        logger.info('[Epoch {}]'.format(epoch_i))

        train_epoch(model, train_data_loader, optimizer, scheduler, scaler, epoch_i, args)
        
        epoch_result = test(model, test_question_data_loader, test_candidate_data_loader, args)

        current_metric = epoch_result[args.main_metric]

        save_file = args.save_dir + f'/model_{args.seed}.pt'
        
        if current_metric > best_metric:
            results[args.seed] = epoch_result
            best_epoch = epoch_i
            best_metric = current_metric
            save_dict = {
                "model_state_dict": model.state_dict(),
            }
            torch.save(save_dict, save_file)
            logger.info('  - [Info] The checkpoint file has been updated.')
    logger.info(f'Got best test performance on epoch{best_epoch}')
    
    logger.info(f'[Seed: {args.seed:d}][Best] MRR ={results[args.seed]["mrr"]: 3.2f} %, '\
                                               f'P@1 ={results[args.seed]["p1"]: 3.2f} %, '\
                                               f'R@5 ={results[args.seed]["r5"]: 3.2f} %, '\
                                               f'R@10 ={results[args.seed]["r10"]: 3.2f} %.')

    if args.rm_saved_model == "True":
        os.remove(save_file)


def prepare_dataloaders(args):
    from utils_data import ReQADataset as Dataset
    # initialize datasets
    train_dataset = Dataset(args, split="train")
    test_question_dataset = Dataset(args, split="test", data_type="question")
    test_candidate_dataset = Dataset(args, split="test", data_type="candidate")
    
    logger.info(f"train data size: {train_dataset.__len__()}")
    logger.info(f"test question size: {test_question_dataset.__len__()}")
    logger.info(f"test candidate size: {test_candidate_dataset.__len__()}")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn)
    
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
    

    return train_data_loader, test_question_data_loader, test_candidate_data_loader


def prepare_model(args):
    from models.dual_encoder import RankModel

    model = RankModel(args)
    model.to(args.device)
    return model


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--overwrite_cache", action="store_true")
    # training
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--mixed_training", action="store_true")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument('--save_results', type=str)
    # evalaution
    parser.add_argument("--main_metric", type=str)
    # model
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--pooler', type=str)
    parser.add_argument("--matching_func", type=str)
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument("--rm_saved_model", type=str, default="True")
    parser.add_argument('--temperature', type=float)

    args = parser.parse_args()

    # set path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger.info(args)
    
    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # record results
    results = dict()
    
    for seed in args.seeds:
        # set seed
        args.seed = seed
        set_seed(args.seed)
        
        # load dataset
        train_data_loader, test_question_data_loader, test_candidate_data_loader = prepare_dataloaders(args)

        # preparing model
        model = prepare_model(args)

        # training
        run(model, train_data_loader, test_question_data_loader, test_candidate_data_loader, args, results)

    
    results["overall"] = defaultdict(dict)
    results["overall"]["mrr"]["mean"] = np.mean([results[seed]["mrr"] for seed in args.seeds])
    results["overall"]["mrr"]["std"] = np.std([results[seed]["mrr"] for seed in args.seeds])
    results["overall"]["p1"]["mean"] = np.mean([results[seed]["p1"] for seed in args.seeds])
    results["overall"]["p1"]["std"] = np.std([results[seed]["p1"] for seed in args.seeds])
    results["overall"]["r5"]["mean"] = np.mean([results[seed]["r5"] for seed in args.seeds])
    results["overall"]["r5"]["std"] = np.std([results[seed]["r5"] for seed in args.seeds])
    results["overall"]["r10"]["mean"] = np.mean([results[seed]["r10"] for seed in args.seeds])
    results["overall"]["r10"]["std"] = np.std([results[seed]["r10"] for seed in args.seeds])

    logger.info(f'[Overall] MRR ={results["overall"]["mrr"]["mean"]: 3.2f} %, '\
                                               f'P@1 ={results["overall"]["p1"]["mean"]: 3.2f} %, '\
                                               f'R@5 ={results["overall"]["r5"]["mean"]: 3.2f} %, '\
                                               f'R@10 ={results["overall"]["r10"]["mean"]: 3.2f} %.')
    
    if args.save_results == "True":
        with open(args.save_dir + "/results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()