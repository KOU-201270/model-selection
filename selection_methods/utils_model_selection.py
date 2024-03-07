import os
import random
import logging
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from rank_bm25 import BM25Okapi as BM25
from openai import OpenAI

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
        # return (vecs - self.mu).dot(self.W)
        return (vecs - np.mean(vecs, axis=0, keepdims=True)).dot(self.W)

    def inverse_transform(self, vecs):
        return vecs.dot(self.inverse_W) + self.mu


def computing_mrr(src_embeddings, tgt_embeddings, candidate_size, args):
    if "dot" in args.method:
        predict_logits = np.sum(src_embeddings * tgt_embeddings, axis=-1).reshape(-1, candidate_size)
    elif "cos" in args.method:
        predict_logits = np.sum((src_embeddings / np.linalg.norm(src_embeddings, axis=-1, keepdims=True)) * \
                                (tgt_embeddings / np.linalg.norm(tgt_embeddings, axis=-1, keepdims=True)), axis=-1).reshape(-1, candidate_size)
    src_size = predict_logits.shape[0]
    labels = [0] * src_size
    
    mrr_counts = 0

    for idx in range(src_size):
        pred = np.argsort(-predict_logits[idx]).tolist()
        # mrr
        for r, p in enumerate(pred):
            if p == labels[idx]:
                mrr_counts += 1 / (r + 1)
                break

    mrr = np.round(mrr_counts / src_size, 4)

    return mrr


def conditional_whitening(src_embeddings, tgt_embeddings, candidate_size):
    
    pca_model = PCA(n_components=src_embeddings.shape[1], whiten=True)\
            .fit(np.concatenate([src_embeddings, tgt_embeddings], 0))
    whiten_src_embeddings, whiten_tgt_embeddings = pca_model.transform(src_embeddings), pca_model.transform(tgt_embeddings)
    pre_mrr = computing_mrr(src_embeddings, tgt_embeddings, candidate_size)
    after_mrr = computing_mrr(whiten_src_embeddings, whiten_tgt_embeddings, candidate_size)

    if after_mrr > pre_mrr:
        return whiten_src_embeddings, whiten_tgt_embeddings
    else:
        return src_embeddings, tgt_embeddings
        




def whitening(src_embeddings, tgt_embeddings):
    pca_model = PCA(n_components=src_embeddings.shape[1], whiten=True)\
        .fit(np.concatenate([src_embeddings, tgt_embeddings], 0))
    return pca_model.transform(src_embeddings), pca_model.transform(tgt_embeddings)


def random_candidate_sampling(src_embeddings, tgt_embeddings, src_ids, args):

    random.seed(args.seed)
    src_size = src_embeddings.shape[0]
    candidate_size = int(args.candidate_size)
    
    # Load data features from cache or datas et file
    cached_dir = f"./cached_data/{args.dataset}"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    cached_file = os.path.join(cached_dir, "random_candidates")
    # load processed dataset or process the original dataset
    if os.path.exists(cached_file):
        logging.info("Loading random candidates from cached file %s", cached_file)
        random_candidates_dict = torch.load(cached_file)
    else:
        random_candidates_dict = dict()
    
    candidate_key = f"{args.seed}_{candidate_size}"
    if candidate_key in random_candidates_dict:
        sampled_candidate_ids = random_candidates_dict[candidate_key]
    else:
        
        equal_matrix = (src_ids[:, np.newaxis].repeat(src_size, 1) == src_ids[np.newaxis, :].repeat(src_size, 0)).astype(np.float32)
        neg_coords = np.where(equal_matrix==0)
        
        neg_ids = []
        cur_row = -1
        sampled_candidate_ids = []
        neg_size = candidate_size - 1
        num_sampling = 0
        for i in range(neg_coords[0].shape[0]):
            if neg_coords[0][i] != cur_row:
                cur_row = neg_coords[0][i]
                if len(neg_ids) > 0:
                    sampled_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
                    num_sampling += 1
                neg_ids.append([])
                
            neg_ids[-1].append(neg_coords[1][i])
        sampled_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))

        random_candidates_dict[candidate_key] = sampled_candidate_ids
        logging.info("Saving candidate ids to %s", cached_file)
        torch.save(random_candidates_dict, cached_file)
    
    sampled_candidate_embeddings = tgt_embeddings[sampled_candidate_ids].reshape(src_size, candidate_size, -1)


    return src_embeddings[:, np.newaxis, :].repeat(candidate_size, 1), sampled_candidate_embeddings

def BM25_candidate_sampling(src_embeddings, tgt_embeddings, src_ids, args):

    src_size = src_embeddings.shape[0]
    candidate_size = int(args.candidate_size)

    # Load data features from cache or datas et file
    cached_dir = f"./cached_data/{args.dataset}"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    cached_file = os.path.join(cached_dir, "BM25_candidates")
    # load processed dataset or process the original dataset
    if os.path.exists(cached_file):
        logging.info("Loading BM25 candidates from cached file %s", cached_file)
        BM25_candidates = torch.load(cached_file)
    else:
        plm_name = [s for s in args.model_name_or_path.split('/') if s !=''][-1]
        cached_dataset_file = os.path.join(cached_dir, f'train_{plm_name}')
        logging.info("Loading dataset from cached file %s", cached_dataset_file)
        data_dict = torch.load(cached_dataset_file)
        questions = data_dict['questions']
        answers = data_dict['answers']

        bm25_corpus = BM25(answers)
        scores = [bm25_corpus.get_scores(question) for question in questions]
        rank = [x[::-1] for x in np.argsort(scores)]

        BM25_candidates = [[y for y in x if src_ids[y] != i] for i, x in enumerate(rank)]
        logging.info("Saving candidate ids to %s", cached_file)
        torch.save(BM25_candidates, cached_file)

    sampled_candidate_ids = []
    neg_size = candidate_size - 1
    for i in range(src_size):
        sampled_candidate_ids.extend([i] + BM25_candidates[i][:neg_size])

    sampled_candidate_embeddings = tgt_embeddings[sampled_candidate_ids].reshape(src_size, candidate_size, -1)


    return src_embeddings[:, np.newaxis, :].repeat(candidate_size, 1), sampled_candidate_embeddings

def ChatGPT_candidate_sampling(src_embeddings, tgt_embeddings, src_ids, args):

    src_size = src_embeddings.shape[0]
    candidate_size = int(args.candidate_size)

    # Load data features from cache or datas et file
    cached_dir = f"./cached_data/{args.dataset}"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    cached_dir = os.path.join(cached_dir, "ChatGPT_candidates")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    client.api_key = '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'

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a professional but careless scientist, always trying to answer the question as presicely and detailed as you can, but still with some subtle mistakes."},
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
      ]
    )


def random_intra_mixup_candidate_sampling(src_embeddings, tgt_embeddings, src_ids, candidate_size, mode, args):
    random.seed(args.seed)
    src_size, embed_size = src_embeddings.shape
    neg_size = candidate_size - 1
    
    # Load data features from cache or datas et file
    cached_dir = f"./cached_data/{args.dataset}"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    cached_file = os.path.join(cached_dir, "random_intra_mixup_candidates")
    # load processed dataset or process the original dataset
    if os.path.exists(cached_file):
        logging.info("Loading random candidates from cached file %s", cached_file)
        candidates_dict = torch.load(cached_file)
    else:
        candidates_dict = dict()
    
    candidate_key = f"{args.seed}_{candidate_size}"
    if candidate_key in candidates_dict:
        sampled_qa_candidate_ids, sampled_aq_candidate_ids = candidates_dict[candidate_key]
    else:
        equal_matrix = (src_ids[:, np.newaxis].repeat(src_size, 1) == src_ids[np.newaxis, :].repeat(src_size, 0)).astype(np.float32)
        neg_coords = np.where(equal_matrix==0)
        
        neg_ids = []
        cur_row = -1
        sampled_qa_candidate_ids = []
        # sampled_qq_candidate_ids = []
        sampled_aq_candidate_ids = []
        # sampled_aa_candidate_ids = []

        num_sampling = 0
        for i in range(neg_coords[0].shape[0]):
            if neg_coords[0][i] != cur_row:
                cur_row = neg_coords[0][i]
                if len(neg_ids) > 0:
                    sampled_qa_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
                    # sampled_qq_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
                    sampled_aq_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
                    # sampled_aa_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
                    num_sampling += 1
                neg_ids.append([])
            neg_ids[-1].append(neg_coords[1][i])

        sampled_qa_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
        # sampled_qq_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
        sampled_aq_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
        # sampled_aa_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))

        candidates_dict[candidate_key] = (sampled_qa_candidate_ids, 
                                        #   sampled_qq_candidate_ids,
                                          sampled_aq_candidate_ids, 
                                        #   sampled_aa_candidate_ids
                                          )
        logging.info("Saving candidate ids to %s", cached_file)
        torch.save(candidates_dict, cached_file)
    
    sampled_qa_candidate_embeddings = tgt_embeddings[sampled_qa_candidate_ids].reshape(src_size, candidate_size, -1)
    # sampled_qq_candidate_embeddings = src_embeddings[sampled_qq_candidate_ids].reshape(src_size, candidate_size, -1)
    sampled_aq_candidate_embeddings = src_embeddings[sampled_aq_candidate_ids].reshape(src_size, candidate_size, -1)
    # sampled_aa_candidate_embeddings = tgt_embeddings[sampled_aa_candidate_ids].reshape(src_size, candidate_size, -1)

    # [batch_size, candidate_size, hidden_size]
    src_embeddings = src_embeddings[:, np.newaxis, :].repeat(candidate_size, 1)
    tgt_embeddings = tgt_embeddings[:, np.newaxis, :].repeat(candidate_size, 1)

    # [batch_size, candidate_size, 1]
    mixup_weight = np.linspace(1, 0, candidate_size)[np.newaxis, :, np.newaxis].repeat(src_size, 0)
    sampled_qa_candidate_embeddings = mixup_weight * tgt_embeddings + (1- mixup_weight) * sampled_qa_candidate_embeddings
    # sampled_qq_candidate_embeddings = mixup_weight * src_embeddings + (1- mixup_weight) * sampled_qq_candidate_embeddings
    sampled_aq_candidate_embeddings = mixup_weight * src_embeddings + (1- mixup_weight) * sampled_aq_candidate_embeddings
    # sampled_aa_candidate_embeddings = mixup_weight * tgt_embeddings + (1- mixup_weight) * sampled_aa_candidate_embeddings

    if mode == "uni":
        src_list = [src_embeddings]
        candidate_list = [sampled_qa_candidate_embeddings]
    elif mode == "bi":
        src_list = [src_embeddings, tgt_embeddings]
        candidate_list = [sampled_qa_candidate_embeddings, \
                        sampled_aq_candidate_embeddings]
    src_embeddings = np.concatenate(src_list, 0)
    sampled_candidate_embeddings = np.concatenate(candidate_list, 0)
    labels = np.concatenate([np.squeeze(mixup_weight, 2).reshape(-1)] * len(candidate_list), 0)

    return src_embeddings.reshape(-1, embed_size), sampled_candidate_embeddings.reshape(-1, embed_size), labels


def random_mixup_candidate_sampling(src_embeddings, tgt_embeddings, src_ids, candidate_size, seed):
    random.seed(seed)
    # qa
    src_size, embed_size = src_embeddings.shape
    equal_matrix = (src_ids[:, np.newaxis].repeat(src_size, 1) == src_ids[np.newaxis, :].repeat(src_size, 0)).astype(np.float32)
    neg_coords = np.where(equal_matrix==0)
    neg_ids = []
    cur_row = -1
    for i in range(neg_coords[0].shape[0]):
        if neg_coords[0][i] != cur_row:
            cur_row = neg_coords[0][i]
            neg_ids.append([])
        neg_ids[-1].append(neg_coords[1][i])
    
    neg_size = candidate_size - 1
    sampled_qa_candidate_ids = []
    sampled_qa_candidate_embeddings = []
    
    for i in range(src_ids.shape[0]):
        sampled_qa_candidate_ids.append([i] + random.sample(neg_ids[i], neg_size))
        qa_candidate_embeddings = [tgt_embeddings[j: j+1] for j in sampled_qa_candidate_ids[-1]]
        sampled_qa_candidate_embeddings.append(np.concatenate(qa_candidate_embeddings, 0)[np.newaxis, :, :])
    
    sampled_qa_candidate_embeddings = np.concatenate(sampled_qa_candidate_embeddings, 0)

    # [batch_size, candidate_size, hidden_size]
    src_embeddings = src_embeddings[:, np.newaxis, :].repeat(candidate_size, 1)
    tgt_embeddings = tgt_embeddings[:, np.newaxis, :].repeat(candidate_size, 1)

    # [batch_size, candidate_size, 1]
    qa_mixup_weight = np.linspace(1, 0, candidate_size)[np.newaxis, :, np.newaxis].repeat(src_size, 0)
    sampled_qa_candidate_embeddings = qa_mixup_weight * tgt_embeddings + (1- qa_mixup_weight) * sampled_qa_candidate_embeddings

    sampled_candidate_embeddings = sampled_qa_candidate_embeddings

    return src_embeddings, sampled_candidate_embeddings


# def intra_candidate_sampling(src_embeddings, tgt_embeddings, src_ids, candidate_size, seed, args):
#     cached_dir = f"./cached_data/{args.dataset}"
#     plm_name = "sentence-transformers/all-mpnet-base-v2"
#     args.model_name_or_path = plm_name
#     cached_dataset_embedding_file = os.path.join(cached_dir, f'{plm_name}_nn_indices')
#     # load processed dataset or process the original dataset
#     if os.path.exists(cached_dataset_embedding_file):
#         logging.info("Loading encoded dataset from cached file %s", cached_dataset_embedding_file)
#         data_dict = torch.load(cached_dataset_embedding_file)
#         qq_topk = data_dict["qq_topk"]
#         aa_logits = data_dict["aa_logits"]
#     else:
#         # load dataset
#         from train_reqa import prepare_dataloaders
#         train_data_loader, test_question_data_loader, test_candidate_data_loader = prepare_dataloaders(args)
#         # preparing model
#         from models.dual_encoder import RankModel
#         model = RankModel(args)
#         model.to(args.device)
#         from model_selection import obtain_train_embeddings
#         train_question_embeddings, train_answer_embeddings, train_question_ids = obtain_train_embeddings(model, train_data_loader)

#         train_question_embeddings = F.normalize(train_question_embeddings, dim=-1)
#         train_answer_embeddings = F.normalize(train_answer_embeddings, dim=-1)
#         qq_logits = train_question_embeddings.mm(train_question_embeddings.t())
#         aa_logits = train_answer_embeddings.mm(train_answer_embeddings.t())
#         batch_size = train_question_embeddings.shape[0]
#         logit_mask = (train_question_ids.unsqueeze(1).repeat(1, batch_size) == train_question_ids.unsqueeze(0).repeat(batch_size, 1)).float()
#         qq_logits -= logit_mask * 100000000
#         aa_logits -= logit_mask * 100000000

#         qq_topk = torch.topk(qq_logits, k=1).indices.cpu()
#         aa_topk = torch.topk(qq_logits, k=1).indices.cpu()

#         saved_data = {
#             "qq_topk": qq_topk,
#             "aa_topk": aa_topk
#         }
                
#         logging.info("Saving encoded dataset to %s", cached_dataset_embedding_file)
#         torch.save(saved_data, cached_dataset_embedding_file)
#     return train_question_embeddings, train_answer_embeddings, train_question_ids, test_question_embeddings, test_candidate_embeddings, test_ground_truths

