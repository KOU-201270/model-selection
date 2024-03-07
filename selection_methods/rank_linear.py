import time
import random
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


class Rank_Reg(object):
    def __init__(self, args):
        self.args = args
    
    def score(self, features, labels):

        start_time = time.time()

        if "scale" in self.args.method:
            scaler = StandardScaler().fit(features)
            features = scaler.transform(features)


        if "balance" in self.args.method:

            pos_ids = np.where(labels==1)[0]
            pos_features = features[pos_ids]
            pos_labels = labels[pos_ids]
            neg_ids = np.where(labels==0)[0]
            neg_ids = random.sample(neg_ids.tolist(), pos_features.shape[0])
            neg_features = features[neg_ids]
            neg_labels = labels[neg_ids]

            train_features = np.concatenate([pos_features, neg_features], 0)
            train_labels = np.concatenate([pos_labels, neg_labels], 0)
            # import pdb; pdb.set_trace()
        else:
            train_features, train_labels = features, labels

        if "lir" in self.args.method:
            model = LinearRegression().fit(train_features, train_labels)
            # import pdb; pdb.set_trace()
            predict_logits = model.predict(features).reshape(-1, int(self.args.candidate_size))

        elif "lor" in self.args.method:
            model = LogisticRegression().fit(train_features, train_labels)
            predict_logits = model.predict_proba(features)[:, 1].reshape(-1, int(self.args.candidate_size))
        # import pdb; pdb.set_trace()

        src_size = predict_logits.shape[0]
        labels = [0] * src_size
        
        pos_prob = 0.0
        p_counts = {1: 0.0}
        r_counts = {5: 0.0, 10: 0.0}
        mrr_counts = 0

        for idx in range(src_size):
            pos_prob += predict_logits[idx, 0] / np.sum(predict_logits[idx])

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
        
        pos_prob /= src_size
        p_at_k = {k: np.round(v / src_size, 4) * 100 for k, v in p_counts.items()}
        r_at_k = {k: np.round(v / src_size, 4) * 100 for k, v in r_counts.items()}
        mrr = np.round(mrr_counts / src_size, 4)

        score = {
            "pos_prob": pos_prob,
            "p1": p_at_k[1],
            "r5": r_at_k[5],
            "r10": r_at_k[10],
            "mrr": mrr
        }

        end_time = time.time()

        return score, end_time - start_time