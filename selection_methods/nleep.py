import time
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale


class NLEEP(object):
    def __init__(self, args):
        self.args = args

    def score(self, X, y, component_ratio=5):
        """
        NLEEP score from https://github.com/TencentARC/SFDA/blob/main/metrics.py

        Args:
            X (_type_): _description_
            y (_type_): _description_
            component_ratio (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """        
        start_time = time.time()
        n = len(y)
        num_classes = len(np.unique(y))

        X = scale(X)

        # GMM: n_components = component_ratio * class number
        n_components_num = component_ratio * num_classes
        gmm = GaussianMixture(n_components=n_components_num, max_iter=1, verbose=1, random_state=self.args.seed).fit(X)
        prob = gmm.predict_proba(X)  # p(z|x)
        
        # NLEEP
        pyz = np.zeros((num_classes, n_components_num))
        for y_ in range(num_classes):
            indices = np.where(y == y_)[0]
            filter_ = np.take(prob, indices, axis=0) 
            pyz[y_] = np.sum(filter_, axis=0) / n   
        pz = np.sum(pyz, axis=0)    
        py_z = pyz / pz             
        py_x = np.dot(prob, py_z.T) 

        # nleep_score
        score = np.sum(py_x[np.arange(n), y]) / n

        end_time = time.time()

        return score, end_time - start_time