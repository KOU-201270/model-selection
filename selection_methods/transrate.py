import time
import numpy as np
from sklearn import preprocessing


def coding_rate(Z, eps=1e-4): 
    n, d = Z.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
    return 0.5 * rate


def transrate(Z, y, eps=1e-4): 
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.
    K = int(y.max() + 1)
    for i in range(K):
        RZY += coding_rate(Z[(y==i).flatten()], eps)
    return RZ - RZY / K


class TransRate(object):
    def __init__(self, args):
        self.args = args
    

    def score(self, features, labels):

        start_time = time.time()
        score = transrate(preprocessing.scale(features), labels)
        end_time = time.time()

        return score, end_time - start_time
