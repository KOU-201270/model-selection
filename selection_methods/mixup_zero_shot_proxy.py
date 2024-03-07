import random
import numpy as np
from selection_methods.method_utils import whitening, random_mixup_candidate_sampling


class ZeroShotProxy(object):
    def __init__(self, args):
        self.args = args

    def matching(self, src_embeddings, tgt_embeddings):

        # if self.matching_func == "cos":
        #     src_embeddings /= np.linalg.norm(src_embeddings, axis=-1, keepdims=True)
        #     tgt_embeddings /= np.linalg.norm(tgt_embeddings, axis=-1, keepdims=True)
        
        predict_logits = np.sum(src_embeddings * tgt_embeddings, axis=-1)

        return predict_logits

    
    def score(self, train_question_embeddings, train_answer_embeddings, train_question_ids, \
                    test_question_embeddings, test_candidate_embeddings, test_ground_truths):
        
        self.candidate_size = int(self.args.method.split("-")[1])
        self.question_size = train_question_embeddings.shape[0]
        if "whitening" in self.args.method:
            train_question_embeddings, train_answer_embeddings = whitening(train_question_embeddings, train_answer_embeddings, self.args.seed)

        repeated_train_question_embeddings, sampled_candidate_embeddings = random_mixup_candidate_sampling(train_question_embeddings, 
                                                                                                     train_answer_embeddings, 
                                                                                                     train_question_ids,
                                                                                                     self.candidate_size, self.args.seed)
        labels = [0] * self.question_size

        # import pdb; pdb.set_trace()

        predict_logits = self.matching(repeated_train_question_embeddings, sampled_candidate_embeddings)

        p1_counts = 0.
        mrr_counts = 0.

        for idx in range(self.question_size):
            pred = np.argsort(-predict_logits[idx]).tolist()

            # precision at K
            if pred[0] == labels[idx]:
                p1_counts += 1
                        
            # mrr
            for r, p in enumerate(pred):
                if p == labels[idx]:
                    mrr_counts += 1 / (r + 1)
                    break
        
        p1 = np.round(p1_counts / self.question_size, 4)
        mrr = np.round(mrr_counts / self.question_size, 4)

        return mrr