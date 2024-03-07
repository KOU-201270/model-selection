import time
import random
import numpy as np
from selection_methods.utils_model_selection import whitening, random_candidate_sampling


class ZeroShotProxyGlobal(object):
    def __init__(self, args):
        self.args = args
    
    # def matching(self, src_embeddings, tgt_embeddings, src_ids=None):

    #     if self.matching_func == "cos":
    #         src_embeddings /= np.linalg.norm(src_embeddings, axis=1, keepdims=True)
    #         tgt_embeddings /= np.linalg.norm(tgt_embeddings, axis=1, keepdims=True)

    #     predict_logits = np.matmul(src_embeddings, tgt_embeddings.T)

    #     if src_ids is not None:
            
            
    #         logit_mask = (src_ids[:, np.newaxis].repeat(self.question_size, 1) == src_ids[np.newaxis, :].repeat(self.question_size, 0)).astype(np.float32) - np.eye(self.question_size)
    #         predict_logits -= logit_mask * 100000000

    #     return predict_logits

    # def matching(self, src_embeddings, tgt_embeddings):

    #     # if self.matching_func == "cos":
    #     #     src_embeddings /= np.linalg.norm(src_embeddings, axis=-1, keepdims=True)
    #     #     tgt_embeddings /= np.linalg.norm(tgt_embeddings, axis=-1, keepdims=True)
        
    #     predict_logits = np.sum(src_embeddings * tgt_embeddings, axis=-1)

    #     return predict_logits


    def score(self, src_features, tgt_features, features, labels):

        start_time = time.time()
    
        predict_logits = predict_logits = np.matmul(src_features, tgt_features.T)
        src_size = predict_logits.shape[0]
        labels = list(range(src_size))
        
        p_counts = {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0, 100: 0.0}
        r_counts = {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0, 100: 0.0}
        mrr_counts = 0

        for idx in range(src_size):
            pred = np.argsort(-predict_logits[idx]).tolist()

            # precision at K
            for rank in p_counts.keys():
                if labels[idx] in pred[:rank]:
                    p_counts[rank] += 1 / rank
            
            # recall at K
            for rank in r_counts.keys():
                if labels[idx] in pred[:rank]:
                    r_counts[rank] += 1
 
            # mrr
            for r, p in enumerate(pred):
                if p == labels[idx]:
                    mrr_counts += 1 / (r + 1)
                    break
        
        p_at_k = {k: np.round(v / src_size, 4) * 100 for k, v in p_counts.items()}
        r_at_k = {k: np.round(v / src_size, 4) * 100 for k, v in r_counts.items()}
        mrr = np.round(mrr_counts / src_size, 4)

        score = {
            "p1": p_at_k[1],
            "p5": p_at_k[5],
            "p10": p_at_k[10],
            "p20": p_at_k[20],
            "p50": p_at_k[50],
            "p100": p_at_k[100],
            "r5": r_at_k[5],
            "r10": r_at_k[10],
            "r20": r_at_k[20],
            "r50": r_at_k[50],
            "r100": r_at_k[100],
            "mrr": mrr
        }

        end_time = time.time()

        return score, end_time - start_time
    
