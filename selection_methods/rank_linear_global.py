import time
import random
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


class Rank_Reg(object):
    def __init__(self, args):
        self.args = args
    
    def predict(self, model, scaler, src_features, tgt_features):
        all_scores = []
        num_sample = src_features.shape[0]
        for i in range(num_sample):
            # import pdb; pdb.set_trace()
            row_features = src_features[i][np.newaxis, :].repeat(num_sample, 0) * tgt_features
            if scaler is not None:
                row_features = scaler.transform(row_features)
            row_scores = model.predict(row_features).reshape(-1)
            all_scores += [row_scores]
        return np.array(all_scores).reshape(num_sample, num_sample)
    
    def score(self, src_features, tgt_features, features, labels):

        start_time = time.time()

        scaler = None
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

        model = LinearRegression().fit(train_features, train_labels)
        predict_logits = self.predict(model, scaler, src_features, tgt_features)
            
        # import pdb; pdb.set_trace()

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