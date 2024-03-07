import time
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Logistic(object):
    def __init__(self, args):
        self.args = args
    
    def score(self, features, labels):

        start_time = time.time()
        
        num_samples = features.shape[0]
        all_index = [i for i in range(num_samples)]
        train_index = random.sample(all_index, int(num_samples * 0.8))
        dev_index = list(set(all_index) - set(train_index))
        train_features = features[train_index]
        train_labels = labels[train_index]
        dev_features = features[dev_index]
        dev_labels = labels[dev_index]

        train_scaler = StandardScaler().fit(train_features)
        train_features = train_scaler.transform(train_features)
        dev_features = train_scaler.transform(dev_features)

        model = LogisticRegression(random_state=self.args.seed, multi_class='multinomial', solver='lbfgs').fit(train_features, train_labels)
        end_time = time.time()

        return (model.predict(dev_features) == dev_labels).mean(), end_time - start_time